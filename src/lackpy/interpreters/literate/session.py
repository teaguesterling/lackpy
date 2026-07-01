"""Multi-round literate fold -- the interpreter-as-fold-returning-Either.

A literate *session* folds model responses across rounds. Each round is one
``step(raw) -> StepResult``:

  - Right (success): the round's document ran cleanly. Returns the clean rendered
    output (``<think>`` reasoning stripped) + advanced state -- the kernel is
    mutated in place and the write journal is committed. ``@continue`` in the doc
    surfaces as ``continue_requested`` so the caller feeds the next round.
  - Left (failure): the round failed. Returns the errors + the *raw* un-interpreted
    message so the model can correct its own work (cleaning would hide the bug).
    State is NOT advanced -- file writes are rolled back (journal) and kernel name
    rebindings are restored.

The session is a PURE fold: ``step`` takes a raw string and returns a result. The
model-call loop (prompt -> model -> step -> feed back) is a thin client's job, so
the fold is testable without an LLM. The client feeds the next round using
``session.scope`` (what is *live*) rather than re-injecting variables as text.

The load-bearing difference from the old ``scripts/literate_agent.py`` loop: a
SINGLE persistent kernel is threaded across rounds, so functions, imports, and
objects defined in one round are genuinely available (as live objects) to the
next -- the old loop rebuilt a fresh kernel each round and passed only
``repr(v)[:200]`` text, which dropped functions entirely.

Left's "state not advanced" is a rebinding-level guarantee: file writes and name
rebindings roll back, but in-place mutations and effects beyond the journal do
not -- cooperative, the same boundary as the effect gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..base import ExecutionContext

# <think>...</think> reasoning blocks (a thinking model's scratch space) must not
# reach the parser -- as prose they would print verbatim into the clean doc and
# pollute the next round's context. Strip closed blocks, then any unclosed trailing
# <think> (model hit the token limit mid-reasoning).
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning from raw model output.

    Handles an unclosed ``<think>`` (truncation mid-reasoning) by stripping to the
    end. Returns the remainder stripped of surrounding whitespace -- which may be
    empty if the model emitted only reasoning (the session treats that as a Left).
    """
    text = _THINK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


@dataclass
class StepResult:
    """The Either returned by :meth:`LiterateSession.step`.

    ``ok`` discriminates: Right carries ``clean_doc`` / ``continue_requested`` /
    ``variables``; Left carries ``errors`` / ``raw`` (the un-interpreted message
    to hand back to the model for correction).
    """

    ok: bool
    clean_doc: str = ""
    continue_requested: bool = False
    variables: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    raw: str = ""


class LiterateSession:
    """A multi-round literate fold over a single persistent kernel.

    Construct with the :class:`ExecutionContext` (tools -> ceiling + namespace,
    ``base_dir`` -> journal). Call :meth:`step` per model response; feed the next
    round while ``result.continue_requested`` is true, and on a Left feed
    ``result.errors`` + ``result.raw`` back for correction.
    """

    def __init__(self, context: ExecutionContext, interpreter: Any = None) -> None:
        # Lazy imports avoid a circular import with the package __init__ (which
        # re-exports this module).
        from . import LiterateInterpreter, _build_namespace
        from .kernel import LightweightKernel

        self._context = context
        self._interpreter = interpreter or LiterateInterpreter()
        self._kernel = LightweightKernel(namespace=_build_namespace(context))
        self._clean_parts: list[str] = []

    async def step(self, raw: str) -> StepResult:
        """Fold one raw model response into the session. See the module docstring."""
        doc = strip_think(raw)
        if not doc:
            # Only reasoning / nothing survived the strip -- a Left (retry), never
            # a silent empty success.
            return StepResult(
                ok=False, raw=raw,
                errors=["empty document after stripping <think> reasoning"],
            )

        # Snapshot BEFORE running so a failed round can undo its name rebindings.
        snapshot = self._kernel.snapshot()
        result = await self._interpreter._run_document(doc, self._context, self._kernel)

        if not result.success:
            # _run_document already rolled the write journal back; restore the
            # kernel's name rebindings so state is not advanced.
            self._kernel.restore(snapshot)
            return StepResult(
                ok=False, raw=raw,
                errors=[result.error or "execution failed"],
            )

        self._clean_parts.append(result.output or "")
        return StepResult(
            ok=True,
            clean_doc=result.output or "",
            continue_requested=bool(result.metadata.get("continue_requested")),
            variables=dict(result.metadata.get("variables", {})),
        )

    @property
    def rendered(self) -> str:
        """The accumulated clean document across all successful rounds."""
        return "".join(self._clean_parts)

    @property
    def scope(self) -> dict[str, str]:
        """What is currently *live* in the kernel (name -> type: repr summary).

        The thin-client feeds this to the model between rounds instead of
        re-injecting variables as text -- the values are already live in the kernel.
        """
        return self._kernel.get_scope()
