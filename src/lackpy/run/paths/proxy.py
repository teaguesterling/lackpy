"""The ``Path`` stand-in: a program's only route to the filesystem (stepy100 Plan 2, Task 4).

This module is part of the sandbox boundary. ``make_path_class(policy)`` gives
a class named ``Path``. Its instances behave as ``pathlib.Path`` for the
members in ``dictionary.PATH_MEMBERS`` and refuse everything else.

- Every I/O member calls ``policy.check`` first: ``"read"`` for the read
  members, ``"write"`` for ``write_text``, ``write_bytes`` and ``mkdir``
  (``mkdir(parents=True)`` checks every directory it creates).
- ``glob``, ``rglob`` and ``iterdir`` omit entries the policy refuses for read,
  and refuse a pattern that is absolute or contains ``..`` (ruling, Task 4).
- A relative path is relative to the workspace, not the process cwd.
  ``str()``, ``os.fspath()`` and ``repr()`` give the path as the program
  built it, as plain ``pathlib`` does. I/O uses the workspace-joined path.
- Every member that returns a path returns a stand-in. Strings, bytes and
  bools are real values.
- No instance holds its real path: ``_STATE`` maps an instance's id to it, and
  a finalizer removes the entry (as ``ModuleProxy`` keeps state off the proxy).
  A stand-in compares equal to an equal stand-in, so a weak-key dictionary,
  which compares keys with ``==``, cannot be used.
- An ``OSError`` names the path as the program built it (``filename``,
  ``filename2`` and so ``str(e)``), as plain Python run from the workspace
  would (review I1). The workspace path never reaches the program this way.
- The class takes the platform class name (``PosixPath`` or ``WindowsPath``),
  so ``type(p).__name__`` and ``repr(type(p))`` match plain Python (ruling I2).
- ``/`` works from both sides, and ``<``, ``<=``, ``>``, ``>=`` order
  stand-ins, as on ``pathlib`` (ruling I3).
- The metaclass refuses every non-dunder class attribute, so ``Path.home()``
  and ``Path.cwd()`` refuse.
"""

from __future__ import annotations

import contextlib
import pathlib
import weakref
from typing import Any, Callable, Iterator

from ...lang import dictionary
from ..refusal import LackpyRefusal, refuse
from .rules import PathPolicy

_STATE: dict[int, pathlib.Path] = {}

_ALLOWED = frozenset(dictionary.PATH_MEMBERS)
_CLASS_NAME = type(pathlib.Path()).__name__
_CLASS_MODULE = type(pathlib.Path()).__module__  # "pathlib._local" on 3.13+


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


class _PathMeta(type):
    def __getattribute__(cls, attr: str) -> Any:
        if _is_dunder(attr):
            return super().__getattribute__(attr)
        refuse(f"Path.{attr}", attr)

    def __setattr__(cls, attr: str, value: Any) -> None:
        refuse(f"Path.{attr} = ...")

    def __delattr__(cls, attr: str) -> None:
        refuse(f"del Path.{attr}")


def make_path_class(policy: PathPolicy) -> type:
    """Return a ``Path`` stand-in class bound to ``policy``."""

    def raw(p: "Path") -> pathlib.Path:
        return _STATE[id(p)]

    def new(path: pathlib.Path) -> "Path":
        obj = object.__new__(Path)
        _STATE[id(obj)] = path
        weakref.finalize(obj, _STATE.pop, id(obj), None)
        return obj

    def unwrap(value: Any) -> Any:
        return raw(value) if isinstance(value, Path) else value

    @contextlib.contextmanager
    def as_built(p: "Path") -> Iterator[None]:
        """Rewrite an OSError's paths from workspace-joined to as-built (review I1)."""
        try:
            yield
        except OSError as e:
            if not raw(p).is_absolute():
                ws = str(policy.workspace)
                for attr in ("filename", "filename2"):
                    value = getattr(e, attr)
                    if isinstance(value, str):
                        if value == ws:
                            setattr(e, attr, ".")
                        elif value.startswith(ws.rstrip("/") + "/"):
                            setattr(e, attr, value[len(ws.rstrip("/")) + 1:])
            raise

    def target(p: "Path", op: str) -> pathlib.Path:
        absolute = policy.absolute(raw(p))
        policy.check(absolute, op)  # type: ignore[arg-type]
        return absolute

    def safe_pattern(pattern: Any, method: str) -> None:
        text = str(unwrap(pattern))
        pure = pathlib.PurePath(text)
        if text.startswith(("/", "\\")) or pure.is_absolute() or ".." in pure.parts:
            refuse(f"Path.{method}({text!r}) (a pattern that is absolute or contains '..')")

    def entries(p: "Path", found: Iterator[pathlib.Path], base: pathlib.Path) -> Iterator["Path"]:
        built = raw(p)
        with as_built(p):
            for entry in found:
                try:
                    policy.check(entry, "read")
                except LackpyRefusal:
                    continue  # ruling: omit, so a deny neither fails the listing nor reveals the entry
                yield new(built / entry.relative_to(base))

    def reader(name: str) -> Callable[..., Any]:
        def member(self: "Path", *args: Any, **kwargs: Any) -> Any:
            absolute = target(self, "read")
            with as_built(self):
                return getattr(absolute, name)(*args, **kwargs)
        member.__name__ = name
        return member

    def writer(name: str) -> Callable[..., Any]:
        def member(self: "Path", *args: Any, **kwargs: Any) -> Any:
            absolute = target(self, "write")
            with as_built(self):
                return getattr(absolute, name)(*args, **kwargs)
        member.__name__ = name
        return member

    class Path(metaclass=_PathMeta):
        __slots__ = ("__weakref__",)

        def __new__(cls, *segments: Any) -> "Path":
            return new(pathlib.Path(*(unwrap(s) for s in segments)))

        def __getattribute__(self, attr: str) -> Any:
            if attr in _ALLOWED or _is_dunder(attr):
                return object.__getattribute__(self, attr)
            refuse(f"Path.{attr}", attr)

        def __setattr__(self, attr: str, value: Any) -> None:
            refuse(f"Path(...).{attr} = ...")

        def __delattr__(self, attr: str) -> None:
            refuse(f"del Path(...).{attr}")

        # Pure members.
        def __truediv__(self, other: Any) -> "Path":
            return new(raw(self) / unwrap(other))

        def __rtruediv__(self, other: Any) -> "Path":
            return new(unwrap(other) / raw(self))

        def __str__(self) -> str:
            return str(raw(self))

        def __fspath__(self) -> str:
            return str(raw(self))

        def __repr__(self) -> str:
            # Not in the brief's member list; without it repr() and printed
            # lists would differ from plain Python (design §5.3).
            return repr(raw(self))

        def __eq__(self, other: object) -> bool:
            if isinstance(other, Path):
                return raw(self) == raw(other)
            if isinstance(other, pathlib.PurePath):
                return raw(self) == other
            return NotImplemented

        def __hash__(self) -> int:
            return hash(raw(self))

        def __lt__(self, other: object) -> bool:
            if isinstance(other, Path):
                return raw(self) < raw(other)
            return NotImplemented

        def __le__(self, other: object) -> bool:
            if isinstance(other, Path):
                return raw(self) <= raw(other)
            return NotImplemented

        def __gt__(self, other: object) -> bool:
            if isinstance(other, Path):
                return raw(self) > raw(other)
            return NotImplemented

        def __ge__(self, other: object) -> bool:
            if isinstance(other, Path):
                return raw(self) >= raw(other)
            return NotImplemented

        @property
        def name(self) -> str:
            return raw(self).name

        @property
        def stem(self) -> str:
            return raw(self).stem

        @property
        def suffix(self) -> str:
            return raw(self).suffix

        @property
        def parts(self) -> tuple[str, ...]:
            return raw(self).parts

        @property
        def parent(self) -> "Path":
            return new(raw(self).parent)

        def with_suffix(self, suffix: str) -> "Path":
            return new(raw(self).with_suffix(suffix))

        def relative_to(self, *others: Any, **kwargs: Any) -> "Path":
            return new(raw(self).relative_to(*(unwrap(o) for o in others), **kwargs))

        def joinpath(self, *others: Any) -> "Path":
            return new(raw(self).joinpath(*(unwrap(o) for o in others)))

        # Read members.
        read_text = reader("read_text")
        read_bytes = reader("read_bytes")
        exists = reader("exists")
        is_file = reader("is_file")
        is_dir = reader("is_dir")

        def glob(self, pattern: Any, **kwargs: Any) -> Iterator["Path"]:
            safe_pattern(pattern, "glob")
            base = target(self, "read")
            with as_built(self):  # 3.13+ may raise here, before iteration (review N1)
                found = base.glob(unwrap(pattern), **kwargs)
            return entries(self, found, base)

        def rglob(self, pattern: Any, **kwargs: Any) -> Iterator["Path"]:
            safe_pattern(pattern, "rglob")
            base = target(self, "read")
            with as_built(self):
                found = base.rglob(unwrap(pattern), **kwargs)
            return entries(self, found, base)

        def iterdir(self) -> Iterator["Path"]:
            base = target(self, "read")
            with as_built(self):  # 3.13+ scandirs eagerly here (review N1)
                found = base.iterdir()
            return entries(self, found, base)

        # Write members.
        write_text = writer("write_text")
        write_bytes = writer("write_bytes")

        def mkdir(self, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
            absolute = target(self, "write")
            if parents:
                for ancestor in absolute.parents:
                    if ancestor.exists():
                        break
                    policy.check(ancestor, "write")
            with as_built(self):
                absolute.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

    type.__setattr__(Path, "__module__", _CLASS_MODULE)
    type.__setattr__(Path, "__name__", _CLASS_NAME)
    type.__setattr__(Path, "__qualname__", _CLASS_NAME)
    return Path
