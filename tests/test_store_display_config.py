import socket
import threading

import pytest

from lpr.config import parse_line
from lpr.display import LcdClient
from lpr.store import PlateStore


def test_store_roundtrip(tmp_path):
    s = PlateStore(tmp_path / "sub" / "p.db")
    s.add("กข1234", "กรุงเทพมหานคร", 0.9, track_id=7, image_path="x.jpg")
    rows = s.latest()
    assert len(rows) == 1 and rows[0][3] == "กข1234" and rows[0][2] == 7


def test_lcd_disabled_and_unreachable_do_not_raise():
    assert LcdClient(None).send("x") is False
    assert LcdClient("127.0.0.1", 1, timeout=0.2).send("x") is False


def test_lcd_sends_utf8():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    got = []

    def serve():
        conn, _ = srv.accept()
        got.append(conn.recv(1024))
        conn.close()

    t = threading.Thread(target=serve)
    t.start()
    assert LcdClient("127.0.0.1", srv.getsockname()[1]).send("กข1234") is True
    t.join(2)
    srv.close()
    assert got[0].decode("utf-8") == "กข1234"


def test_parse_line():
    assert parse_line("0.1,0.75,0.9,0.75") == (0.1, 0.75, 0.9, 0.75)
    for bad in ("1,2", "a,b,c,d", "0,0,0,0", "0,0,2,1"):
        with pytest.raises(ValueError):
            parse_line(bad)


def test_mask_url_hides_password_only():
    from lpr.config import mask_url

    assert mask_url("rtsp://admin:Secr3t!@192.168.1.5:554/video") == "rtsp://admin:***@192.168.1.5:554/video"
    assert mask_url("rtsp://192.168.1.5/video") == "rtsp://192.168.1.5/video"
    assert mask_url("video.mp4") == "video.mp4"
