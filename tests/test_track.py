from lpr.track import LineCounter, Track, side_of_line

A, B = (0, 100), (200, 100)  # horizontal line at y=100


def car(track_id, cy):
    return Track(track_id, (90, cy - 10, 110, cy + 10))


def test_side_of_line():
    assert side_of_line((50, 50), A, B) != side_of_line((50, 150), A, B)
    assert side_of_line((50, 100), A, B) == 0


def test_fires_once_when_crossing():
    c = LineCounter(A, B)
    assert not c.crossed(car(1, 50))
    assert not c.crossed(car(1, 80))
    assert c.crossed(car(1, 130))      # jumped over the line between frames
    assert not c.crossed(car(1, 160))  # never fires twice
    assert not c.crossed(car(1, 50))


def test_no_fire_without_crossing_and_ids_independent():
    c = LineCounter(A, B)
    assert not c.crossed(car(1, 50))
    assert not c.crossed(car(2, 150))
    assert not c.crossed(car(1, 60))
    assert c.crossed(car(2, 60))
