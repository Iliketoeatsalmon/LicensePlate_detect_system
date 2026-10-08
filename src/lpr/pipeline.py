"""Live pipeline: camera -> cars -> track -> line crossing -> crop -> plate -> DB + LCD."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import cv2

from .config import VEHICLE_CLASSES, Settings, mask_url
from .display import LcdClient
from .overlay import draw_lines_top_right, load_font
from .plate import PlateReader
from .store import PlateStore
from .track import LineCounter, Tracker

log = logging.getLogger(__name__)

CAR_CONF = 0.5
MAX_RECONNECTS = 10


class SetupError(Exception):
    """Something the user must fix before the system can run (message says what)."""


def check_models(settings: Settings) -> None:
    missing = [p for p in (settings.car_model, settings.plate_model, settings.char_model) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise SetupError(f"Model file(s) not found: {names}. Re-clone the repo or set MODELS_DIR.")


def load_models(settings: Settings):
    from ultralytics import YOLO

    check_models(settings)
    return (
        YOLO(str(settings.car_model)),
        PlateReader(YOLO(str(settings.plate_model)), YOLO(str(settings.char_model))),
    )


def open_capture(source: str):
    if not source:
        raise SetupError("No camera set. Put RTSP_URL=rtsp://user:pass@CAMERA_IP:554/... in .env, "
                         "or run: lpr run --source <rtsp-url | video file | 0>")
    target = int(source) if source.isdigit() else source
    if isinstance(target, str) and target.startswith("rtsp://"):
        # TCP is more reliable than UDP; time out (5 s) instead of hanging on a wrong IP.
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000")
    cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG) if isinstance(target, str) else cv2.VideoCapture(target)
    if not cap.isOpened():
        raise SetupError(f"Cannot open video source {mask_url(source)!r}. Check the IP/URL, username/password and network.")
    return cap


def _is_live(source: str) -> bool:
    return source.isdigit() or "://" in source


def detect_cars(car_model, frame, device):
    out = []
    for res in car_model(frame, conf=CAR_CONF, classes=list(VEHICLE_CLASSES), device=device, verbose=False):
        for box in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append((x1, y1, x2, y2, float(box.conf[0])))
    return out


def run(settings: Settings, max_frames: int | None = None) -> int:
    """Run until the source ends, 'q' is pressed or Ctrl+C. Returns the number of plates saved."""
    car_model, reader = load_models(settings)
    cap = open_capture(settings.source)
    ok, frame = cap.read()
    if not ok:
        raise SetupError(f"Connected to {mask_url(settings.source)!r} but could not read a frame.")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = settings.line
    a, b = (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h))

    tracker, counter = Tracker(), LineCounter(a, b)
    store, lcd = PlateStore(settings.db_path), LcdClient(settings.lcd_host, settings.lcd_port)
    settings.crop_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(settings.font_path)
    recent: list[str] = []
    saved = frames = reconnects = 0
    log.info("Running on %s (device=%s). Press q in the window or Ctrl+C to stop.", mask_url(settings.source), settings.device)

    try:
        while ok:
            frames += 1
            tracks = tracker.update(frame, detect_cars(car_model, frame, settings.device))
            for t in tracks:
                if not counter.crossed(t):
                    continue
                tx1, ty1, tx2, ty2 = t.bbox
                crop = frame[max(ty1, 0):ty2, max(tx1, 0):tx2]
                if crop.size == 0:
                    continue
                result = reader.read(crop)
                if result is None:
                    log.info("car #%s crossed the line, no plate read", t.track_id)
                    continue
                path = settings.crop_dir / f"car_{t.track_id}_{int(time.time())}.jpg"
                cv2.imwrite(str(path), crop)
                store.add(result.plate, result.province, result.confidence, t.track_id, path)
                lcd.send(f"{result.plate} {result.province}".strip())
                saved += 1
                recent = ([f"{result.plate} {result.province}".strip()] + recent)[:4]
                log.info("car #%s -> %s %s (%.2f)", t.track_id, result.plate, result.province, result.confidence)

            if settings.show:
                for t in tracks:
                    cv2.rectangle(frame, t.bbox[:2], t.bbox[2:], (0, 255, 0), 2)
                    cv2.putText(frame, f"#{t.track_id}", (t.bbox[0], max(t.bbox[1] - 8, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.line(frame, a, b, (0, 0, 255), 2)
                shown = draw_lines_top_right(frame, recent, font) if recent else frame
                cv2.imshow("LicensePDet (q to quit)", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and frames >= max_frames:
                break

            ok, frame = cap.read()
            while not ok and _is_live(settings.source) and reconnects < MAX_RECONNECTS:
                reconnects += 1
                log.warning("Lost the camera, reconnecting (%d/%d)...", reconnects, MAX_RECONNECTS)
                cap.release()
                time.sleep(min(2 * reconnects, 10))
                cap = cv2.VideoCapture(settings.source, cv2.CAP_FFMPEG)
                ok, frame = cap.read()
            if ok:
                reconnects = 0
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        store.close()
        if settings.show:
            cv2.destroyAllWindows()
    log.info("Stopped. %d frames, %d plates saved to %s", frames, saved, settings.db_path)
    return saved


def read_images(settings: Settings, paths: list[Path]) -> list[tuple[Path, object]]:
    from ultralytics import YOLO

    for p in (settings.plate_model, settings.char_model):
        if not p.exists():
            raise SetupError(f"Model file not found: {p}")
    reader = PlateReader(YOLO(str(settings.plate_model)), YOLO(str(settings.char_model)))
    results = []
    for path in paths:
        img = cv2.imread(str(path))
        results.append((path, reader.read(img) if img is not None else None))
    return results
