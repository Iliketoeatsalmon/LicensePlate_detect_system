"""SQLite log of every plate read."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS plates (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    seen_at     TEXT    NOT NULL,
    track_id    INTEGER,
    plate       TEXT    NOT NULL,
    province    TEXT    NOT NULL DEFAULT '',
    confidence  REAL,
    image_path  TEXT
)
"""


class PlateStore:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def add(self, plate, province="", confidence=None, track_id=None, image_path=None, seen_at=None) -> int:
        seen_at = seen_at or datetime.now().isoformat(timespec="seconds")
        cur = self.conn.execute(
            "INSERT INTO plates (seen_at, track_id, plate, province, confidence, image_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (seen_at, track_id, plate, province, confidence, str(image_path) if image_path else None),
        )
        self.conn.commit()
        return cur.lastrowid

    def latest(self, limit=20):
        cur = self.conn.execute(
            "SELECT id, seen_at, track_id, plate, province, confidence, image_path "
            "FROM plates ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def close(self):
        self.conn.close()
