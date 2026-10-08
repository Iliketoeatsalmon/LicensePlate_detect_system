# Restructure and complete the LicensePlate detect system — design

Status: implemented (2026-10-09). Differences from the draft below: PR 2 and 3 were delivered together in one
PR; model weights stay in plain git (no LFS) so a plain `git clone` works; `mars-small128.pb` was dropped
because `deep-sort-realtime` ships its own embedder; the counting line is configured as fractions of the frame (`LINE`).

## Goal
Make the repo professional (clean, documented, runnable on macOS/Windows/Linux) and finish the
unfinished features: line-crossing car crop, SQLite logging, LCD output.

## Current state (findings)
- Thai license plate reader: YOLO11 finds cars, DeepSORT tracks them, two YOLO models find the
  plate and read its characters (`lpr.py`, `functions/charecter.py`).
- `Deepsort/deep_sort` is a broken gitlink (no `.gitmodules`), so `functions/tracker.py` cannot import.
- `carfollowip.py` computes the distance to a counting line but never uses it.
- `lcd.py` is a manual TCP client to a Raspberry Pi (`192.168.1.101:12345`); `sqllite.py` is empty.
- Hard-coded `/Users/sarawit/...` paths, `.to("cuda")`, and an RTSP URL containing a password.
- Junk tracked in git: `.DS_Store`, `__pycache__`, `tempCodeRunnerFile.py`, `image copy N.png`.
- Province table (`charecter.py`) has suspicious entries: `BTG`, `MSN`, `NPT` (dup of `NBP`),
  `CNT` (dup of `CRI`). Report to owner before changing; the model may emit these codes.

## Target layout
```
README.md  requirements.txt  .gitignore  .env.example
src/lpr/  config.py detect.py track.py plate.py thai.py store.py display.py cli.py
models/   (large weights, see Open decisions)
tests/    thai, plate assembly, store, display (fake socket)
docs/
```

## Data flow
camera/video -> YOLO (car) -> DeepSORT (ID) -> first line crossing -> crop -> plate read ->
SQLite row (id, time, plate, province, confidence, image path) -> send plate text to LCD.

## Behaviour
- Config via `.env` (RTSP_URL, DEVICE, model paths, LCD_HOST/PORT). No secrets in code.
- Device auto-select: CUDA -> MPS -> CPU.
- Camera drop: reconnect with backoff. LCD unreachable: log and continue.
- Tracker: replace the vendored DeepSORT with `deep-sort-realtime` (pip); `mars-small128.pb` dropped.

## Delivery (separate PRs, feature branches, never direct to main)
1. Cleanup: `.gitignore`, remove junk, secrets to `.env`, README, requirements.
2. Restructure into `src/lpr/`, fix misspelled names (`deteection`, `charecter`, `sqllite`), add tests, swap DeepSORT.
3. Features: line-crossing crop, SQLite, LCD.

## Testing
pytest for logic that needs no camera/GPU; smoke test `cli read-images` on `CarCrops` images.
Anything not run is reported as "not verified".

## Open decisions (temporary defaults, owner to confirm)
- Model weights (~70 MB): kept in plain git so `git clone` just works (LFS would need `git lfs` installed).
- Repo visibility: unknown. If public, rotate the camera password (it is in git history).
  History rewrite needs a force push and happens only on explicit owner instruction.
