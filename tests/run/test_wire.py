"""The tagged-JSON wire between the service and a worker (Plan 3, Task 3)."""

import ast
import socket
from pathlib import Path

import pytest

from lackpy.run import wire
from lackpy.run.wire import Reader, WireError, decode, encode, send

_NESTED = {"a": [(1, {2, 3}), b"\x00", {1: frozenset({"x"})}], "t": ({"__t": "x"},)}


@pytest.mark.parametrize("value", [
    None, True, 3, 2.5, "s", b"\x00\xff", (1, 2), {1, 2}, frozenset({"a"}),
    {1: "a"}, {"__t": "x"}, [], (), {}, _NESTED,
    [(1, 2), [b"b", {"k": (None,)}]], {(1, 2): "tuple key"},
])
def test_round_trip(value):
    line = encode(value)
    assert "\n" not in line
    back = decode(line)
    assert back == value
    assert type(back) is type(value)


def test_round_trip_keeps_nested_types():
    back = decode(encode(_NESTED))
    assert type(back["a"][0]) is tuple and type(back["a"][0][1]) is set
    assert type(back["a"][1]) is bytes
    assert type(back["a"][2]) is dict and type(back["a"][2][1]) is frozenset
    assert type(back["t"]) is tuple and type(back["t"][0]) is dict


def test_nested_frozenset_is_not_decoded_as_a_set():
    back = decode(encode([frozenset({"x"}), {"k": (b"b", {1})}]))
    assert type(back[0]) is frozenset
    assert type(back[1]["k"]) is tuple and type(back[1]["k"][0]) is bytes
    assert type(back[1]["k"][1]) is set


def test_bool_stays_bool_and_int_stays_int():
    assert decode(encode(True)) is True
    assert decode(encode(1)) == 1 and type(decode(encode(1))) is int


def test_unknown_tag_rejected():
    with pytest.raises(WireError):
        decode('{"__t": "nope", "v": 1}')


def test_malformed_line_is_a_wire_error():
    with pytest.raises(WireError):
        decode("{not json")


@pytest.mark.parametrize("value", [object(), len, 1j, bytearray(b"x"), {"a": object()},
                                   [len], type("S", (str,), {})("x")])
def test_non_data_rejected(value):
    with pytest.raises(WireError):
        encode(value)


def test_cycle_is_rejected():
    x: list = []
    x.append(x)
    with pytest.raises(WireError):
        encode(x)


def test_reader_eof_timeout_and_cap():
    a, b = socket.socketpair()
    try:
        reader = Reader(b)
        with pytest.raises(TimeoutError):
            reader.recv(0.05)
        send(a, {"op": "x", "v": (1, 2)})
        send(a, {"op": "y"})
        assert reader.recv(1) == {"op": "x", "v": (1, 2)}
        assert reader.recv(1) == {"op": "y"}
        a.close()
        assert reader.recv(1) is None
    finally:
        a.close()
        b.close()


def test_reader_timeout_is_one_deadline_not_per_recv():
    import threading
    import time
    a, b = socket.socketpair()
    stop = threading.Event()

    def dribble():
        while not stop.is_set():
            try:
                a.sendall(b"x")
            except OSError:
                return
            time.sleep(0.05)

    t = threading.Thread(target=dribble, daemon=True)
    t.start()
    try:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            Reader(b).recv(timeout=0.3)
        assert time.monotonic() - start < 0.5
    finally:
        stop.set()
        t.join()
        a.close()
        b.close()


def test_reader_caps_a_line_that_never_ends():
    a, b = socket.socketpair()
    try:
        reader = Reader(b, max_line=100)
        a.sendall(b"x" * 500)
        with pytest.raises(WireError):
            reader.recv(1)
    finally:
        a.close()
        b.close()


def test_reader_caps_a_long_complete_line():
    a, b = socket.socketpair()
    try:
        reader = Reader(b, max_line=100)
        a.sendall(b'"' + b"x" * 200 + b'"\n')
        with pytest.raises(WireError):
            reader.recv(1)
    finally:
        a.close()
        b.close()


def test_reader_truncated_line_at_eof_is_a_wire_error():
    a, b = socket.socketpair()
    try:
        a.sendall(b'{"op": ')
        a.close()
        with pytest.raises(WireError):
            Reader(b).recv(1)
    finally:
        b.close()


def test_channel_imports_no_pickle():
    """Gate 6: no module on the service-worker channel can unpickle."""
    run_dir = Path(wire.__file__).parent
    names = ["wire.py", "zygote.py", "process.py", "limits.py", "session_worker.py",
             "sessions.py"]
    found = [run_dir / n for n in names if (run_dir / n).exists()]
    assert run_dir / "wire.py" in found
    banned = {"pickle", "marshal", "shelve"}
    for path in found:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                mods = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                mods = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (mods & banned), f"{path.name} imports {mods & banned}"
