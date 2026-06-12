"""Best-effort recording of ``delegate()`` results into blq's invocation database.

When a workspace has a ``.bird/`` (i.e. blq is in use there) and ``blq-cli`` is
installed, each delegation is written as a blq invocation under the ``delegate``
source — so delegations become queryable alongside build/test runs (``blq history``,
``blq query``, ``blq output``). This replaces the old ``.lackpy/traces.jsonl`` sink
with blq's "bird invocation format".

Entirely best-effort: a missing blq, no ``.bird/``, or a concurrent writer holding
the single-writer DuckDB lock all result in a silent no-op. Recording must never
affect the delegation result.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def record_delegation(workspace: Path, intent: str, result: dict[str, Any]) -> str | None:
    """Record a ``delegate()`` result as a blq invocation.

    Returns the blq run id, or ``None`` when nothing was recorded — blq not
    installed, the workspace has no ``.bird/``, or any error (e.g. a concurrent
    blq writer holds the DuckDB lock). Never raises.
    """
    try:
        from blq.storage import BlqStorage
    except ImportError:
        return None

    bird_dir = Path(workspace) / ".bird"
    if not bird_dir.is_dir():
        return None

    try:
        run_meta, events, output = _to_invocation(bird_dir, intent, result)
        storage = BlqStorage.open(bird_dir)
        try:
            return storage.write_run(run_meta, events=events, output=output)
        finally:
            try:
                storage.close()
            except Exception:
                pass
    except Exception:
        # A concurrent blq writer (single-writer DuckDB), a schema mismatch, etc.
        # must never break delegation. Swallow and move on.
        return None


def _to_invocation(
    bird_dir: Path, intent: str, result: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, bytes | None]:
    """Map a delegate() result dict onto blq's (run_meta, events, output) shape."""
    total_ms = result.get("total_time_ms") or 0
    completed = datetime.now()
    started = completed - timedelta(milliseconds=total_ms)
    success = bool(result.get("success"))

    run_meta = {
        "command": f'lackpy delegate "{intent[:200]}"',
        "source_name": "delegate",
        "source_type": "import",
        "tag": "delegate",
        "exit_code": 0 if success else 1,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "cwd": str(bird_dir.parent),
        # write_run forwards `environment` (JSON) but NOT `extension_data`, so the
        # structured delegation summary rides in `environment.lackpy`.
        "environment": {
            "lackpy": {
                "success": success,
                "grade": result.get("grade"),
                "generation_tier": result.get("generation_tier"),
                "generation_time_ms": result.get("generation_time_ms"),
                "execution_time_ms": result.get("execution_time_ms"),
                "total_time_ms": total_ms,
                "tools": [e.get("tool") for e in result.get("trace") or []],
                "files_read": result.get("files_read"),
                "files_modified": result.get("files_modified"),
                "correction_attempts": result.get("correction_attempts"),
            }
        },
    }

    # Only failures are diagnostics — one error event so `events(severity="error")`
    # surfaces failed delegations. Successful runs add no events.
    events = None
    if not success:
        events = [{
            "severity": "error",
            "event_type": "delegation_failure",
            "message": result.get("error") or "delegation failed",
            "tool_name": "lackpy",
            "category": "delegate",
        }]

    # The generated program is the artifact worth eyeballing via `blq output`.
    program = result.get("program") or ""
    output = program.encode("utf-8") if program else None

    return run_meta, events, output
