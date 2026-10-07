"""Loopback HTTP server for replay fetching tests."""

from __future__ import annotations

import threading
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Self, cast

Response = tuple[int, bytes]


class _ReplayRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        status, body = cast(ReplayHTTPServer, self.server).response(self.path)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class ReplayHTTPServer(ThreadingHTTPServer):
    """Serve each path's response sequence, repeating its last response on later requests."""

    def __init__(self, responses: dict[str, tuple[Response, ...]]) -> None:
        if any(not sequence for sequence in responses.values()):
            raise ValueError("Each HTTP path requires at least one response")
        super().__init__(("127.0.0.1", 0), _ReplayRequestHandler)
        self._responses = {path: deque(sequence) for path, sequence in responses.items()}
        self._calls: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def calls(self) -> Counter[str]:
        with self._lock:
            return self._calls.copy()

    def response(self, path: str) -> Response:
        with self._lock:
            self._calls[path] += 1
            sequence = self._responses.get(path)
            if sequence is None:
                return 500, b"No response configured for this path"
            return sequence.popleft() if len(sequence) > 1 else sequence[0]

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join()
