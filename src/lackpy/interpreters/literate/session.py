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

STATELESSNESS CONTRACT (the canonical mode): the writer holds NO hidden
conversational state. Every round it is handed a fresh prompt containing the
current document view, and the session folds its raw emission into the one
persistent kernel. The document (plus the kernel state it produced) is the sole
source of truth -- renders append/annotate, never silently rewrite authored
content. One consequence: a stateless writer, shown the current view, may begin
its emission by RE-ECHOING the tail of that view. Without a guard the echo is
re-parsed -- re-executing its cells (harmful for non-idempotent ones) and
re-printing its prose. :func:`strip_overlap` is that guard: it cuts an emission
prefix that matches a suffix of the shown view, before parsing.
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


#: Minimum echoed-prefix length (chars) before strip_overlap cuts. Short
#: coincidental matches (a shared ".\n", a stray word) must never strip.
_MIN_OVERLAP = 8


def strip_overlap(shown: str, emission: str, *, min_overlap: int = _MIN_OVERLAP) -> str:
    """Cut a stateless writer's re-echo of the document view it was shown.

    Finds the LONGEST suffix of ``shown`` that is a prefix of ``emission`` and
    returns ``emission`` with that prefix removed. Everything the writer was
    shown has already been parsed/executed/rendered -- re-parsing the echo would
    re-print prose and re-execute cells (non-idempotent cells corrupt state).

    Guards against false positives:
      - the overlap must be at least ``min_overlap`` characters, and
      - must contain non-whitespace (a run of shared newlines is not an echo).

    A legitimate emission that *coincidentally* opens with >=``min_overlap``
    characters equal to the view's tail is still stripped -- inherent to a
    text-level guard; the cut applies only at the very start of the emission,
    never mid-document.

    Trailing whitespace of ``shown`` is ignored when matching: renders append
    trailing newlines the writer never echoes (and ``strip_think`` has already
    stripped the emission's leading whitespace), so requiring them would make
    the guard miss nearly every real echo.

    The descending scan is O(len(shown) * len(emission)) in the pathological
    worst case (highly repetitive text); the common big case -- the writer
    re-echoes the ENTIRE view -- matches on the first comparison.
    """
    shown = shown.rstrip()
    limit = min(len(shown), len(emission))
    for size in range(limit, min_overlap - 1, -1):
        if emission.startswith(shown[-size:]):
            if not emission[:size].strip():
                # Whitespace-only match; every shorter match is a prefix of
                # this one, hence also whitespace-only. Nothing to strip.
                return emission
            return emission[size:]
    return emission


#: The writer-controlled pause marker. A COMPLETE fenced cell
#: (```lackpy @continue ... ```) pauses via the compiler sentinel; the textual
#: fallback below catches the marker BEFORE a complete fence exists.
CONTINUE_MARKER = "@continue"

_FENCE_LINE_RE = re.compile(r"^```(\S.*)?\s*$")
_FENCE_CLOSE_RE = re.compile(r"^```\s*$")


def split_at_continue(doc: str) -> tuple[str, bool]:
    """Textual fallback for the writer-controlled pause: cut at ``@continue``.

    The compiler-sentinel path requires a COMPLETE fenced ``@continue`` cell.
    A writer that signals the pause mid-emission -- a bare ``@continue`` line,
    or an emission cut right at the marker by a client-side stop sequence --
    never produces that fence. This fallback cuts the document at the FIRST
    such marker and DISCARDS the remainder: content past a pause request is
    reasoning-without-values and is protocol-correct to drop (the writer asked
    to see values before continuing; anything it wrote after the ask was
    written without them).

    Returns ``(kept_document, continue_requested)``. Handled shapes:

      - a bare ``@continue`` line outside any fence -> cut before it;
      - a trailing UNCLOSED ````lackpy @continue`` fence-open (stop-sequence
        cut right after the marker) -> drop the dangling open line;
      - a bare ``@continue`` as the trailing content of an unclosed lackpy
        fence (stop-sequence cut inside an open cell) -> drop the partial
        cell whole rather than auto-close and execute a half-written cell.

    NOT handled (on purpose): a complete fenced ``@continue`` cell (the
    sentinel path owns it) and ``@continue`` inside a *closed* fence body
    (that is code; static analysis reports it).
    """
    lines = doc.split("\n")
    in_fence = False
    fence_info = ""
    fence_open_idx = -1

    for i, line in enumerate(lines):
        if in_fence:
            if _FENCE_CLOSE_RE.match(line):
                in_fence = False
            continue
        if line.strip() == CONTINUE_MARKER:
            return "\n".join(lines[:i]).rstrip("\n"), True
        m = _FENCE_LINE_RE.match(line)
        if m:
            in_fence = True
            fence_info = (m.group(1) or "").strip()
            fence_open_idx = i

    if in_fence and fence_info.startswith("lackpy"):
        if CONTINUE_MARKER in fence_info:
            # Cut landed on the fence-open line itself: "```lackpy @continue".
            return "\n".join(lines[:fence_open_idx]).rstrip("\n"), True
        tail = [ln for ln in lines[fence_open_idx + 1:] if ln.strip()]
        if tail and tail[-1].strip() == CONTINUE_MARKER:
            return "\n".join(lines[:fence_open_idx]).rstrip("\n"), True

    return doc, False


class StopScanner:
    """Client-side stop-sequence scanner for a streaming model call.

    Feed the raw stream chunk by chunk; when a stop sequence appears, ``feed``
    returns True, :attr:`text` holds everything up to AND INCLUDING the
    matched marker, and the caller should abort the stream (closing the HTTP
    connection makes Ollama cancel generation).

    Why client-side rather than the API-native ``options.stop``:

      1. Native stop STRIPS the matched text from the response, and
         ``done_reason`` cannot distinguish a stop-sequence hit from a natural
         end of turn -- the pause marker (the semantic payload) would be
         silently lost, so the pause intent would be unrecoverable.
      2. Native stop matches inside ``<think>`` reasoning blocks, so a
         thinking model *musing* about ``@continue`` would be cut
         mid-reasoning. This scanner suppresses matches inside think blocks.

    Keeping the marker means the downstream textual fallback
    (:func:`split_at_continue`) owns the pause semantics -- one source of
    truth for both streaming and non-streaming clients.

    Rescans the post-think region on each feed -- O(total^2) for pathological
    chunk counts, fine for harness-scale responses.
    """

    def __init__(self, stops: list[str]) -> None:
        self._stops = [s for s in stops if s]
        self._text = ""
        self._stopped = False

    @property
    def text(self) -> str:
        """Accumulated text; cut at (and including) the marker once stopped."""
        return self._text

    @property
    def stopped(self) -> bool:
        return self._stopped

    def feed(self, chunk: str) -> bool:
        """Accumulate ``chunk``; return True once a stop sequence has fired."""
        if self._stopped:
            return True
        self._text += chunk
        start = self._scan_start()
        if start is None:
            return False
        region = self._text[start:]
        best: tuple[int, str] | None = None
        for stop in self._stops:
            idx = region.find(stop)
            if idx != -1 and (best is None or idx < best[0]):
                best = (idx, stop)
        if best is not None:
            idx, stop = best
            self._text = self._text[: start + idx + len(stop)]
            self._stopped = True
            return True
        return False

    def _scan_start(self) -> int | None:
        """Start of the scannable region: after the close of the most recent
        ``<think>`` block; None while one is still open (suppress matching)."""
        last_open = self._text.rfind("<think")
        if last_open == -1:
            return 0
        close = self._text.find("</think>", last_open)
        if close == -1:
            return None
        return close + len("</think>")


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

    async def step(self, raw: str, shown: str | None = None) -> StepResult:
        """Fold one raw model response into the session. See the module docstring.

        Args:
            raw: The writer's raw emission for this round.
            shown: The exact document view the writer was prompted with, for
                the overlap-strip guard (see :func:`strip_overlap`). Defaults
                to the session's accumulated ``rendered`` -- correct for a thin
                client that shows the writer the rendered document (a suffix
                match against the last round's output is a suffix match
                against ``rendered``). Pass explicitly when the client shows
                something else (e.g. the raw document source).
        """
        doc = strip_think(raw)
        if not doc:
            # Only reasoning / nothing survived the strip -- a Left (retry), never
            # a silent empty success.
            return StepResult(
                ok=False, raw=raw,
                errors=["empty document after stripping <think> reasoning"],
            )

        # Overlap-strip guard: cut the re-echoed tail of the shown view before
        # parsing, so echoed cells are not re-executed nor echoed prose
        # re-printed. Statelessness is the canonical mode (module docstring).
        view = self.rendered if shown is None else shown
        if view:
            doc = strip_overlap(view, doc)
            if not doc.strip():
                return StepResult(
                    ok=False, raw=raw,
                    errors=[
                        "emission was only a re-echo of the shown document "
                        "(no new content after overlap-strip)"
                    ],
                )

        # Writer-controlled pause, textual fallback: a bare `@continue` (or an
        # emission cut at the marker by a client-side stop sequence) pauses at
        # the first occurrence; the remainder is discarded (see
        # split_at_continue). The fenced form still pauses via the sentinel.
        doc, marker_pause = split_at_continue(doc)
        if marker_pause and not doc.strip():
            # The emission was ONLY the pause marker -- an empty round that
            # asks for values back, same Right shape as a lone fenced cell.
            return StepResult(ok=True, clean_doc="", continue_requested=True)

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
            continue_requested=(
                bool(result.metadata.get("continue_requested")) or marker_pause
            ),
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
