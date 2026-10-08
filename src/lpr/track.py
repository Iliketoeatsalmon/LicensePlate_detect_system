"""Vehicle tracking (DeepSORT) and line-crossing detection."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2


class Tracker:
    def __init__(self, max_age: int = 30):
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self._ds = DeepSort(max_age=max_age)

    def update(self, frame, detections) -> list[Track]:
        """detections: list of (x1, y1, x2, y2, score)."""
        raw = [([x1, y1, x2 - x1, y2 - y1], score, "vehicle") for x1, y1, x2, y2, score in detections]
        out = []
        for t in self._ds.update_tracks(raw, frame=frame):
            if not t.is_confirmed() or t.time_since_update > 1:
                continue
            x1, y1, x2, y2 = (int(v) for v in t.to_ltrb())
            out.append(Track(int(t.track_id), (x1, y1, x2, y2)))
        return out


def side_of_line(point, a, b) -> int:
    """-1 / 0 / +1: which side of the line a->b the point is on."""
    cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
    return (cross > 0) - (cross < 0)


class LineCounter:
    """Fires once per track id, the first time its center moves from one side of the line to the other."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self._side: dict[int, int] = {}
        self._fired: set[int] = set()

    def crossed(self, track: Track) -> bool:
        side = side_of_line(track.center, self.a, self.b)
        prev = self._side.get(track.track_id)
        self._side[track.track_id] = side or (prev or 0)
        if track.track_id in self._fired or prev is None or side == 0:
            return False
        if prev != 0 and side != prev:
            self._fired.add(track.track_id)
            return True
        return False
