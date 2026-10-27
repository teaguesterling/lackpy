"""The service-worker wire: tagged JSON, one message per line (Plan 3, Task 3).

Only data crosses the socket: the types ``runner._is_data`` accepts (None, bool,
int, float, str, bytes, and list, tuple, set, dict of those). JSON has no bytes,
tuple, set or non-str dict key, so those carry a ``{"__t": tag, ...}`` wrapper.
Nothing here imports ``pickle``, ``marshal`` or ``shelve`` (gate 6): a worker
that is compromised can send a bad message, never code the service will run.
"""

from __future__ import annotations

import base64
import json
import socket
import time
from typing import Any

_DATA_SCALARS = (type(None), bool, int, float, str)
DEFAULT_MAX_LINE = 16 * 1024 * 1024


class WireError(Exception):
    """A value that is not wire data, or a malformed or oversize message."""


def _enc(v: Any, on_path: set[int]) -> Any:
    kind = type(v)
    if kind in _DATA_SCALARS:
        return v
    if kind is bytes:
        return {"__t": "bytes", "v": base64.b64encode(v).decode("ascii")}
    if kind not in (list, tuple, set, frozenset, dict):
        raise WireError(f"{kind.__name__} is not wire data")
    key = id(v)
    if key in on_path:
        raise WireError("a container that contains itself is not wire data")
    on_path.add(key)
    try:
        if kind is list:
            return [_enc(i, on_path) for i in v]
        if kind is dict:
            if all(type(k) is str for k in v) and "__t" not in v:
                return {k: _enc(i, on_path) for k, i in v.items()}
            return {"__t": "dict",
                    "v": [[_enc(k, on_path), _enc(i, on_path)] for k, i in v.items()]}
        tag = {tuple: "tuple", set: "set", frozenset: "frozenset"}[kind]
        return {"__t": tag, "v": [_enc(i, on_path) for i in v]}
    finally:
        on_path.discard(key)


def encode(obj: Any) -> str:
    """One line of JSON for ``obj``, no newline. Non-data raises ``WireError``."""
    try:
        return json.dumps(_enc(obj, set()), separators=(",", ":"), ensure_ascii=True)
    except (RecursionError, ValueError, TypeError) as e:
        raise WireError(f"cannot encode: {type(e).__name__}") from None


def _dec(v: Any) -> Any:
    if isinstance(v, list):
        return [_dec(i) for i in v]
    if not isinstance(v, dict):
        return v
    if "__t" not in v:
        return {k: _dec(i) for k, i in v.items()}
    tag, body = v["__t"], v.get("v")
    if tag == "bytes" and isinstance(body, str):
        return base64.b64decode(body.encode("ascii"), validate=True)
    if tag in ("tuple", "set", "frozenset") and isinstance(body, list):
        items = [_dec(i) for i in body]
        return {"tuple": tuple, "set": set, "frozenset": frozenset}[tag](items)
    if tag == "dict" and isinstance(body, list):
        return {_dec(k): _dec(i) for k, i in body}
    raise WireError(f"unknown or malformed tag {tag!r}")


def decode(line: str) -> Any:
    """The value ``encode`` made ``line`` from. Bad input raises ``WireError``."""
    try:
        return _dec(json.loads(line))
    except WireError:
        raise
    except (ValueError, TypeError, RecursionError) as e:  # JSON, base64, unhashable key
        raise WireError(f"cannot decode: {type(e).__name__}") from None


def send(sock: socket.socket, msg: dict[str, Any]) -> None:
    """Write ``msg`` as one line."""
    sock.sendall((encode(msg) + "\n").encode("ascii"))


class Reader:
    """Reads one message per line from ``sock``, holding at most ``max_line`` bytes of a line."""

    def __init__(self, sock: socket.socket, max_line: int = DEFAULT_MAX_LINE) -> None:
        self._sock = sock
        self._max = max_line
        self._buf = bytearray()

    def recv(self, timeout: float | None) -> dict[str, Any] | None:
        """The next message; None at EOF; ``TimeoutError`` at ``timeout`` seconds."""
        old = self._sock.gettimeout()
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                end = self._buf.find(b"\n")
                if end >= 0:
                    if end > self._max:
                        raise WireError(f"line longer than {self._max} bytes")
                    line = bytes(self._buf[:end])
                    del self._buf[:end + 1]
                    msg = decode(line.decode("ascii", errors="replace"))
                    if not isinstance(msg, dict):
                        raise WireError("a message must be a dict")
                    return msg
                if len(self._buf) > self._max:
                    raise WireError(f"line longer than {self._max} bytes")
                if deadline is None:
                    self._sock.settimeout(None)
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for a message")
                    self._sock.settimeout(remaining)
                chunk = self._sock.recv(min(65536, self._max + 1 - len(self._buf)))
                if not chunk:
                    if self._buf:
                        raise WireError("connection closed in the middle of a line")
                    return None
                self._buf += chunk
        finally:
            self._sock.settimeout(old)
