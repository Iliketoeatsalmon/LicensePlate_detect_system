"""Read a Thai license plate from a car crop (logic moved from the old lpr.py)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .thai import data_province, get_thai_character, split_license_plate_and_province

CONF_PLATE = 0.30
CONF_CHAR = 0.35
LINE_Y_TOL = 30  # px; raise if the camera perspective is strong


@dataclass(frozen=True)
class PlateResult:
    plate: str
    province: str
    confidence: float
    box: tuple[int, int, int, int]  # plate box inside the crop

    @property
    def letters(self) -> str:
        return "".join(ch for ch in self.plate if not ch.isdigit())

    @property
    def digits(self) -> str:
        return "".join(ch for ch in self.plate if ch.isdigit())


def group_by_line(tokens, y_tol=LINE_Y_TOL):
    """Group character tokens (x1, cls, box) into lines by Y, each sorted left->right."""
    if not tokens:
        return []
    tmp = [(x1, cls, box, (box[1] + box[3]) / 2.0) for x1, cls, box in tokens]
    tmp.sort(key=lambda t: t[3])
    lines = []
    for item in tmp:
        for line in lines:
            if abs(item[3] - np.mean([it[3] for it in line])) <= y_tol:
                line.append(item)
                break
        else:
            lines.append([item])
    for line in lines:
        line.sort(key=lambda t: t[0])
    return [[(x1, cls, box) for (x1, cls, box, _) in line] for line in lines]


def assemble_plate_and_province(lines):
    """Build (plate_text, province_text) from grouped lines (cars: 1 row, motorcycles: 2)."""
    if not lines:
        return "", ""

    if len(lines) == 1:
        flat = lines[0]
        non_prov = [c for (_, c, _) in flat if c not in data_province]
        prov = [c for (_, c, _) in flat if c in data_province]
        plate_text = "".join(get_thai_character(c) for c in non_prov)
        joined = "".join(get_thai_character(c) for c in non_prov + prov)
        _, province_text = split_license_plate_and_province(joined)
        return plate_text, province_text or ""

    scored = []
    for idx, line in enumerate(lines):
        non_p = sum(1 for (_, c, _) in line if c not in data_province)
        p = sum(1 for (_, c, _) in line if c in data_province)
        scored.append((idx, non_p, -p))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    plate_idx = scored[0][0]

    plate_line = lines[plate_idx]
    others = [ln for i, ln in enumerate(lines) if i != plate_idx]
    province_line = max(others, key=lambda ln: sum(1 for (_, c, _) in ln if c in data_province)) if others else []

    plate_text = "".join(get_thai_character(c) for (_, c, _) in plate_line if c not in data_province)
    province_text = "".join(get_thai_character(c) for (_, c, _) in province_line if c in data_province)
    if not province_text:
        combined = "".join(get_thai_character(c) for (_, c, _) in plate_line + province_line)
        _, alt = split_license_plate_and_province(combined)
        province_text = alt or ""
    return plate_text, province_text


class PlateReader:
    """Two YOLO models: one finds the plate in a car crop, one reads its characters."""

    def __init__(self, plate_model, char_model):
        self.plate_model = plate_model
        self.char_model = char_model

    def read(self, car_bgr) -> PlateResult | None:
        found = self.plate_model(car_bgr, conf=CONF_PLATE, verbose=False)
        best = None
        for res in found:
            for box in res.boxes:
                conf = float(box.conf[0])
                if best is None or conf > best[0]:
                    best = (conf, tuple(int(v) for v in box.xyxy[0]))
        if best is None:
            return None

        conf, (x1, y1, x2, y2) = best
        roi = car_bgr[max(y1, 0):y2, max(x1, 0):x2]
        if roi.size == 0:
            return None

        tokens, confs = [], []
        for res in self.char_model(roi, conf=CONF_CHAR, verbose=False):
            for cbox in res.boxes:
                cx1, cy1, cx2, cy2 = (int(v) for v in cbox.xyxy[0])
                name = self.char_model.names[int(cbox.cls[0])]
                tokens.append((cx1, name, (cx1, cy1, cx2, cy2)))
                confs.append(float(cbox.conf[0]))

        plate, province = assemble_plate_and_province(group_by_line(tokens))
        if not plate:
            return None
        mean_conf = float(np.mean(confs)) if confs else conf
        return PlateResult(plate=plate, province=province, confidence=mean_conf, box=(x1, y1, x2, y2))
