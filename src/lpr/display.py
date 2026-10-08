"""Send plate text to the LCD (a Raspberry Pi listening on a TCP port). Never raises."""
from __future__ import annotations

import logging
import socket

log = logging.getLogger(__name__)


class LcdClient:
    def __init__(self, host: str | None, port: int = 12345, timeout: float = 2.0):
        self.host, self.port, self.timeout = host, port, timeout

    @property
    def enabled(self) -> bool:
        return bool(self.host)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout) as s:
                s.sendall(text.encode("utf-8"))
            return True
        except OSError as exc:
            log.warning("LCD %s:%s unreachable (%s) - continuing without it", self.host, self.port, exc)
            return False
