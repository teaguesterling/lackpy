"""Transactional file journal for the literate batch path.

Snapshots a set of files before a document's write-cells run, so the whole
document's *statically known, literal-path* writes can be rolled back atomically
if any cell fails. Only paths the effect classifier surfaced as ``writes``
(literal-arg ``@write``/``@diff``/``write_file``/``apply_diff`` targets) are
journaled; dynamic-path writes, exec (``run_command``), and unanalyzable cells
are NOT covered -- their effects can't be bounded statically. Pair with the
ceiling gate (which refuses exec/unanalyzable cells under a low ceiling) to get
full atomicity for a policy-governed run.

The transaction boundary here is the whole batch ``execute()`` call. Per-cell
``@continue`` commit points are a streaming/fold concern for a later slice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .tools import _resolve as _resolve_tool_path


class FileJournal:
    """Records pre-execution file contents so writes can be undone on failure.

    For each tracked path the journal stores either its bytes (file existed) or
    ``None`` (file did not exist). ``rollback`` restores those bytes or deletes a
    file that was created during the transaction; ``commit`` drops the snapshots.

    The journal resolves paths through the SAME function and base the literate
    write tools use (``tools._resolve``), so a snapshot path is, by construction,
    the exact path ``write_file``/``apply_diff`` will touch -- they cannot drift
    into snapshotting one file while the write lands on another.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        # Mirror make_tool_namespace: base is None when unset (the tools then
        # resolve relative paths against the cwd, and so do we -- via the shared
        # _resolve, at the same call sites within one execute()).
        self._base = Path(base_dir) if base_dir else None
        self._snapshots: dict[Path, bytes | None] = {}

    def _resolve(self, path: str) -> Path:
        return _resolve_tool_path(path, self._base)

    def snapshot(self, paths: Iterable[str]) -> None:
        """Record the current state of each path (idempotent per path)."""
        for path in paths:
            rp = self._resolve(path)
            if rp in self._snapshots:
                continue
            self._snapshots[rp] = rp.read_bytes() if rp.is_file() else None

    def rollback(self) -> None:
        """Restore every tracked path to its snapshotted state."""
        for rp, content in self._snapshots.items():
            if content is None:
                # The run created it -> remove it. Guard against a directory
                # sharing the path (unlink on a dir raises): only files/symlinks.
                if rp.is_file() or rp.is_symlink():
                    rp.unlink()
            else:
                rp.parent.mkdir(parents=True, exist_ok=True)
                rp.write_bytes(content)

    def commit(self) -> None:
        """Accept the writes -- drop the snapshots so nothing can be rolled back."""
        self._snapshots.clear()

    @property
    def tracked(self) -> int:
        return len(self._snapshots)
