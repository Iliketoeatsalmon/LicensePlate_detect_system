"""Command line: lpr run | read-images | history | check"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import ROOT, load_settings, mask_url

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lpr", description="Thai license plate recognition")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="read plates from a camera / video (default: RTSP_URL in .env)")
    run.add_argument("--source", help="rtsp URL, video file or camera index (overrides .env)")
    run.add_argument("--no-show", action="store_true", help="do not open a preview window")
    run.add_argument("--max-frames", type=int, help="stop after N frames (for testing)")

    ri = sub.add_parser("read-images", help="read plates from image files / folders")
    ri.add_argument("paths", nargs="*", type=Path, default=[ROOT / "samples"])

    hist = sub.add_parser("history", help="show the latest saved plates")
    hist.add_argument("-n", type=int, default=20)

    sub.add_parser("check", help="check models, device and camera before running")
    return p


def _cmd_check(settings) -> int:
    from .pipeline import SetupError, check_models, open_capture

    print(f"device : {settings.device}")
    print(f"source : {mask_url(settings.source) or '(not set)'}")
    try:
        check_models(settings)
        print("models : OK")
        cap = open_capture(settings.source)
        ok, _ = cap.read()
        cap.release()
        print("camera : OK" if ok else "camera : connected but no frame")
        return 0 if ok else 1
    except SetupError as exc:
        print(f"problem: {exc}")
        return 1


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args = _build_parser().parse_args(argv)
    from .pipeline import SetupError

    try:
        if args.cmd == "run":
            from .pipeline import run

            settings = load_settings(args.source, show=False if args.no_show else None)
            run(settings, max_frames=args.max_frames)
            return 0

        settings = load_settings()
        if args.cmd == "check":
            return _cmd_check(settings)
        if args.cmd == "history":
            from .store import PlateStore

            for row in PlateStore(settings.db_path).latest(args.n):
                print(" | ".join("-" if v is None else str(v) for v in row))
            return 0
        if args.cmd == "read-images":
            from .pipeline import read_images

            files: list[Path] = []
            for p in args.paths:
                files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT) if p.is_dir() else [p]
            for path, res in read_images(settings, files):
                print(f"{path.name}: " + (f"{res.plate} | {res.province or '-'} | conf {res.confidence:.2f}" if res else "no plate"))
            return 0
    except SetupError as exc:
        print(f"\nCannot start: {exc}", file=sys.stderr)
        return 2
    return 1
