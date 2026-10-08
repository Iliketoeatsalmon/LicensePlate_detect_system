# LicensePDet — Thai License Plate Detection

ระบบตรวจจับรถและอ่านป้ายทะเบียนไทยจากกล้องวงจรปิด
CCTV-based Thai license plate recognition: detect vehicles, track them, read the plate.

## How it works / การทำงาน

```
กล้อง RTSP / วิดีโอ → YOLO11 หารถ → DeepSORT ให้ ID → crop รถ
                    → YOLO หาป้าย → YOLO อ่านตัวอักษร → ป้ายทะเบียน + จังหวัด
```

| Part | File | Status |
|------|------|--------|
| Car detection + tracking | `cctv-traffic/detect-car/carfollowip.py`, `carfollowcrop.py` | needs DeepSORT (see below) |
| Plate reading | `cctv-traffic/detect-car/lpr.py`, `functions/charecter.py` | working on sample images |
| LCD output (Raspberry Pi) | `cctv-traffic/detect-car/lcd.py` | manual only, not connected yet |
| SQLite logging | `cctv-traffic/detect-car/sqllite.py` | empty, planned |

## Setup

```bash
conda create -n lpr python=3.10 -y && conda activate lpr
pip install -r requirements.txt
cp .env.example .env      # then fill in RTSP_URL etc. — never commit .env
```

## Run

Read plates from the sample images (run from `license_plate_deteection/`):

```bash
cd license_plate_deteection
PYTHONPATH=cctv-traffic/detect-car python cctv-traffic/detect-car/lpr.py
```

Press a key to move to the next image.

## Known issues

- `cctv-traffic/detect-car/Deepsort/deep_sort` is an empty folder (broken submodule), so the tracker
  scripts cannot import it yet. It will be replaced by the `deep-sort-realtime` package.
- Paths in `lpr.py` are relative to the folder you run it from.
- Some province codes in `functions/charecter.py` look wrong (`BTG`, `MSN`, duplicate `NPT`/`CNT`).

## Roadmap

1. Cleanup (this PR)
2. Restructure into `src/lpr/`, fix misspelled names, add tests, replace DeepSORT
3. Line-crossing crop, SQLite logging, LCD output

Design and plans: `docs/superpowers/`
