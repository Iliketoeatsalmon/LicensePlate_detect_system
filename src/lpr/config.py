"""Settings, read from environment variables (and a .env file if present)."""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

VEHICLE_CLASSES = (2, 3, 5, 7)  # COCO: car, motorcycle, bus, truck


def parse_line(text: str) -> tuple[float, float, float, float]:
    """'x1,y1,x2,y2' as fractions (0-1) of the frame width/height."""
    try:
        x1, y1, x2, y2 = (float(v) for v in text.split(","))
    except ValueError as exc:
        raise ValueError(f"LINE must be 'x1,y1,x2,y2' (fractions 0-1), got {text!r}") from exc
    if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
        raise ValueError(f"LINE values must be between 0 and 1, got {text!r}")
    if (x1, y1) == (x2, y2):
        raise ValueError("LINE start and end points must differ")
    return x1, y1, x2, y2


def mask_url(url: str) -> str:
    """rtsp://user:secret@host/... -> rtsp://user:***@host/... (for logs and messages)."""
    return re.sub(r"(://[^:/@\s]+):[^@/\s]+@", r"\1:***@", url)


def pick_device(want: str = "auto") -> str:
    want = want.lower()
    if want != "auto":
        return want
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_show() -> bool:
    return sys.platform in ("darwin", "win32") or bool(os.environ.get("DISPLAY"))


@dataclass(frozen=True)
class Settings:
    source: str
    device: str
    line: tuple[float, float, float, float]
    db_path: Path
    crop_dir: Path
    models_dir: Path
    font_path: Path
    lcd_host: str | None
    lcd_port: int
    show: bool

    @property
    def car_model(self) -> Path:
        return self.models_dir / "yolo11n.pt"

    @property
    def plate_model(self) -> Path:
        return self.models_dir / "license_plate.pt"

    @property
    def char_model(self) -> Path:
        return self.models_dir / "data_plate.pt"


def load_settings(source: str | None = None, show: bool | None = None) -> Settings:
    load_dotenv(ROOT / ".env")
    env = os.environ.get
    return Settings(
        source=source or env("SOURCE") or env("RTSP_URL") or "",
        device=pick_device(env("DEVICE", "auto")),
        line=parse_line(env("LINE", "0.1,0.75,0.9,0.75")),
        db_path=ROOT / env("DB_PATH", "data/plates.db"),
        crop_dir=ROOT / env("CROP_DIR", "data/crops"),
        models_dir=ROOT / env("MODELS_DIR", "models"),
        font_path=ROOT / "assets" / "THK2D.ttf",
        lcd_host=env("LCD_HOST") or None,
        lcd_port=int(env("LCD_PORT", "12345")),
        show=_default_show() if show is None else show,
    )
