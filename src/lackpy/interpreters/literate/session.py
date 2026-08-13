"""Multi-round literate fold -- the interpreter-as-fold-returning-Either.

A literate *session* folds model responses across rounds. Each round is one
``step(raw) -> StepResult``, an Either under the TWO-LAYER forgiveness
contract (L1.1/L1.2 — design conflict #2, option (c)):

  - Right (success): the round's document ran cleanly. Returns the clean rendered
    output (``<think>`` reasoning stripped) + advanced state -- the kernel is
    mutated in place. ``@continue`` in the doc surfaces as
    ``continue_requested`` so the caller feeds the next round.
  - Left (failure): the round contained at least one REIFIED failure — an
    unknown name bound as a typed hole (``hole_opened``) or a runtime error
    bound as an error value (``error_reified``). The Left is an AGGREGATE
    VIEW derived from the session ledger's per-round slice, NOT a
    control-flow abort: the round still completes, its output joins
    ``rendered`` (annotated through the [kernel] channel), and its partial
    state is KEPT as real values + forgiving values — nothing rolls back.
    Returns the ledger-derived errors + the *raw* un-interpreted message so
    the model can correct its own work (cleaning would hide the bug); a
    corrected round simply rebinds the holes with real values.
  - A pre-execution refusal (parse errors — since L1.5 the effect-ceiling
    gate reifies per cell as ``source_unavailable`` instead of refusing the
    document) is also a Left, but nothing ran and no state changed.

The session is a PURE fold: ``step`` takes a raw string and returns a result. The
model-call loop (prompt -> model -> step -> feed back) is a thin client's job, so
the fold is testable without an LLM. The client feeds the next round using
``session.scope`` (what is *live*) rather than re-injecting variables as text.

The load-bearing difference from the old ``scripts/literate_agent.py`` loop: a
SINGLE persistent kernel is threaded across rounds, so functions, imports, and
objects defined in one round are genuinely available (as live objects) to the
next -- the old loop rebuilt a fresh kernel each round and passed only
``repr(v)[:200]`` text, which dropped functions entirely.

Historical note: before L1.1/L1.2 a Left was an abort — the write journal
rolled file writes back and the kernel's name rebindings were restored. That
rollback contract is retired (deliberately, with its tests): forgiveness keeps
the work and records the failure in the queryable session ledger instead.

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
from .annotations import session_manifest
from .parser import to_compute_tags, to_markdown

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
# The same pause in <compute> form. Both are live: the tag is the documented
# syntax, fences remain accepted input, and a client streaming either one needs
# a stop sequence that matches what its prompt asked for.
COMPUTE_CONTINUE_MARKER = "<compute continue>"
_COMPUTE_CLOSE = "</compute>"

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
        if line.strip() == COMPUTE_CONTINUE_MARKER:
            # A CLOSED tag is a complete cell: the parser normalises it and the
            # compiler sentinel owns the pause, exactly as for a closed fence.
            # Only a dangling open tag -- the stop-scanner cut shape -- is ours.
            if _COMPUTE_CLOSE not in "\n".join(lines[i + 1:]):
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

    WHICH ARTIFACT DO YOU WANT? A session emits four documents, and choosing the
    wrong one fails silently rather than loudly — the caller gets *a* document,
    just not the one meant for that reader:

    ==========================  ============================================
    for the WRITER, next round  :attr:`continued` (= :attr:`rendered`)
    for a READER               :attr:`display`
    for publishing elsewhere   :attr:`markdown`
    for a notebook             ``to_notebook()`` in ``kernel.formats``
    ==========================  ============================================

    :attr:`continued` preserves authored source (in the syntax the writer used)
    plus the ``[kernel]`` channel, and re-parses cleanly when fed back.
    :attr:`display` is the executed output — expanded prose and printed values,
    no source. Feeding :attr:`display` back poisons the loop: flat stdout
    re-prints and stacks on re-emission. ``result.clean_doc`` is one round of
    :attr:`display`; for a multi-round document it is the wrong scope.
    """

    def __init__(
        self,
        context: ExecutionContext,
        interpreter: Any = None,
        *,
        max_rounds: int | None = None,
        session_id: str | None = None,
        document_id: str | None = None,
        persistence: Any = None,
    ) -> None:
        """``session_id`` / ``document_id`` (optional) are the identity
        stamped on every ledger row AND every serialized binding version this
        session produces; when not given, the ledger auto-generates a session
        id (today's behavior) and the binding history shares it.

        ``persistence`` (optional) is a
        :class:`~.kernel.persistence.PersistenceBackend`: when given, the
        session's ledger/versions are the persistence-backed subclasses, so
        every recorded row and asserted binding version is ALSO handed
        (serialized) to the backend.  When ``None`` — the default — the plain
        in-memory classes are used and kernel behavior is byte-identical to a
        session without persistence wiring at all (the no-op-when-unplugged
        guarantee)."""
        # Lazy imports avoid a circular import with the package __init__ (which
        # re-exports this module).
        from . import LiterateInterpreter, _build_namespace
        from .kernel import (
            BindingVersions,
            Ledger,
            LightweightKernel,
            PersistentBindingVersions,
            PersistentLedger,
        )

        self._context = context
        # L4: the client loop's REAL round cap (max_iterations), surfaced to the
        # writer through the per-splice manifest. None = no cap configured. The
        # session does not enforce it — the client loop does — it makes the
        # budget VISIBLE (the writer otherwise never sees max_iterations).
        self._max_rounds = max_rounds
        # L4 segment counter: writer emissions folded so far. One step() = one
        # writer call = one budget unit (mirrors the client's for-loop over
        # max_iterations, which spends an iteration on Lefts and marker-only
        # pause rounds too — so those consume here as well).
        self._segments = 0
        self._interpreter = interpreter or LiterateInterpreter()
        self._kernel = LightweightKernel(
            namespace=_build_namespace(context), base_dir=context.base_dir
        )
        # ONE ledger threaded across every round (L1.0/L1.1): the queryable
        # record of the session's execution + reified failures. Each step's
        # aggregate Either is derived from this ledger, per-round slice.
        # ONE binding version history threaded across every round (L1.3):
        # bindings are immutable versions; re-asserting a name in a later
        # round supersedes the prior version and is ledgered. The kernel's
        # namespace dict stays the mutable latest-wins surface.
        # Identity: the versions history SHARES the ledger's session id (which
        # is the caller's when given, else the ledger's generated one), so
        # serialized ledger rows and binding versions carry the same identity.
        if persistence is not None:
            self._ledger = PersistentLedger(
                persistence, session_id=session_id, document_id=document_id
            )
            self._versions = PersistentBindingVersions(
                persistence,
                session_id=self._ledger.session_id,
                document_id=document_id,
            )
        else:
            self._ledger = Ledger(session_id=session_id, document_id=document_id)
            self._versions = BindingVersions(
                session_id=self._ledger.session_id, document_id=document_id
            )
        # Flat stdout per round (printed values, `@scratch` summaries): the
        # `clean_doc` view used by the Left-correction path.
        self._clean_parts: list[str] = []
        # Canonical SOURCE-PRESERVING render per round (authored prose + code
        # source + ledger-derived [kernel] notes): the `rendered` view fed back
        # to a stateless writer. Kept separate from `_clean_parts` so the
        # feedback loop is round-trippable (exp1 close) while `clean_doc` stays
        # the flat "what printed this round" view.
        self._rendered_parts: list[str] = []
        # L7 startup self-test: BEFORE this session accepts any work, the
        # kernel proves it can evaluate (known value, hole, error, unavailable,
        # round-trip) — see selftest.run_selftest.  The outcome is observable
        # (the `selftest` property) and ledgered; on failure the kernel is
        # disabled and every step() returns the structured refusal (fail
        # closed: a session whose kernel cannot evaluate must refuse, not
        # produce garbage).  The probe battery uses a throwaway ledger, so
        # this session ledger records only the one summary entry.
        from .selftest import SELFTEST_FAILED, SELFTEST_PASSED, run_selftest

        self._selftest = run_selftest(self._kernel, self._interpreter)
        if self._selftest.ok:
            self._ledger.record(
                SELFTEST_PASSED,
                detail={"probes": [p.probe for p in self._selftest.probes]},
            )
        else:
            self._ledger.record(SELFTEST_FAILED, detail=self._selftest.summary())

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
        # L7 fail-closed refusal: a session whose startup self-test failed
        # refuses EVERY work attempt with the legible structured refusal
        # (headline + which probe failed, expected vs got) — before any
        # bookkeeping, because a dead session folds nothing.
        if not self._selftest.ok:
            return StepResult(
                ok=False, raw=raw, errors=self._selftest.refusal_lines()
            )

        # L4 budget consumption: every fold of a writer emission — including
        # ones that end in a Left or a marker-only pause — consumed one model
        # call in the client loop, so it consumes one budget unit here.
        self._segments += 1

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

        from .kernel.forgiveness import describe_failure, reified_failures

        # No snapshot/rollback (L1.1/L1.2 contract change): failures during a
        # round reify as bound values (holes / error values) and the round
        # completes; a pre-execution refusal never runs a cell, so there is
        # nothing to restore either way.
        round_mark = len(self._ledger)
        result = await self._interpreter._run_document(
            doc, self._context, self._kernel,
            ledger=self._ledger, versions=self._versions,
        )

        if not result.metadata.get("completed"):
            # Pre-execution refusal (parse errors; the ceiling gate reifies
            # per cell since L1.5 and completes instead): no cell ran, no
            # state changed — a plain Left with the refusal reason.
            return StepResult(
                ok=False, raw=raw,
                errors=[result.error or "execution failed"],
            )

        # The round COMPLETED — its output and bindings stand, even when
        # failures were reified. The Either below is the aggregate view
        # derived from this round's ledger slice (conflict #2 option (c)).
        self._clean_parts.append(result.output or "")
        # Accumulate the canonical source-preserving render for this round —
        # what `rendered` (the fed-back document) exposes. Derived in
        # `_run_document` from this round's cells + ledger slice. L4: every
        # splice OPENS with the session manifest, a [kernel] block (inert on
        # reparse, covered by strip-stale like any channel note) reporting the
        # segment index, the remaining pause budget (cap − segments consumed,
        # from the client's real max_iterations), and the observation count
        # drawn from the queryable ledger at render time (post-round — the
        # count covers everything delivered THROUGH this splice).
        manifest = session_manifest(
            segment=self._segments,
            observations=len(self._ledger),
            cap=self._max_rounds,
        )
        # Spell the round-trip artifact the way the writer spelled it. Parsing
        # normalises <compute> to fences, so without this a tag-authored
        # document comes back as fences on the next pause and the writer
        # switches syntax mid-conversation -- back to the form that truncates a
        # payload containing a fence, which is the whole reason for the tag.
        rendered_markdown = result.metadata.get("rendered_markdown", "")
        if "<compute" in raw:
            rendered_markdown = to_compute_tags(rendered_markdown)
        self._rendered_parts.append(manifest + "\n\n" + rendered_markdown)
        continue_requested = (
            bool(result.metadata.get("continue_requested")) or marker_pause
        )
        variables = dict(result.metadata.get("variables", {}))

        failures = reified_failures(self._ledger.entries()[round_mark:])
        if failures:
            return StepResult(
                ok=False,
                raw=raw,
                errors=[describe_failure(e) for e in failures],
                clean_doc=result.output or "",
                continue_requested=continue_requested,
                variables=variables,
            )
        return StepResult(
            ok=True,
            clean_doc=result.output or "",
            continue_requested=continue_requested,
            variables=variables,
        )

    @property
    def selftest(self):
        """The L7 startup self-test report (:class:`~.selftest.SelfTestReport`)
        — the observable liveness status.  ``selftest.ok`` False means the
        kernel is disabled and every :meth:`step` refuses with
        ``selftest.refusal_lines()``."""
        return self._selftest

    @property
    def ledger(self):
        """The session's queryable event ledger (L1.0) — one across all
        rounds. Reified failures (``hole_opened`` / ``error_reified``) live
        here; the per-round Left is derived from it, never thrown."""
        return self._ledger

    @property
    def segments_consumed(self) -> int:
        """L4: writer emissions folded so far (one per :meth:`step` call —
        Lefts and marker-only pause rounds consumed a model call too). The
        manifest's segment index is this counter at splice time."""
        return self._segments

    @property
    def max_rounds(self) -> int | None:
        """L4: the client loop's round cap (``max_iterations``) as configured,
        or ``None`` when the client set no cap. Surfaced, not enforced."""
        return self._max_rounds

    @property
    def budget_remaining(self) -> int | None:
        """L4: ``max_rounds - segments_consumed`` (``None`` when uncapped) —
        the same arithmetic the manifest prints at each splice."""
        if self._max_rounds is None:
            return None
        return self._max_rounds - self._segments

    @property
    def versions(self):
        """The session's binding version history (L1.3) — one across all
        rounds. Every name a cell binds is an immutable
        :class:`~.kernel.versions.BindingVersion`; re-assertion supersedes
        (``superseded_by`` + a ``superseded`` ledger entry), so prior
        versions are never lost even though the exec namespace is
        latest-wins."""
        return self._versions

    @property
    def rendered(self) -> str:
        """The canonical SOURCE-PRESERVING document across all completed rounds
        (including aggregate-Left rounds — their source stands, annotated).

        This is the exp1 close: what a thin client feeds back to the stateless
        writer is THIS render — authored prose + code source preserved, with
        each round's forgiveness state (holes / error values / unavailable
        sources / supersessions) drawn FROM THE LEDGER and routed through the
        inert ``[kernel]`` channel — NOT the flat stdout (``clean_doc``). Fed
        back and re-emitted, it re-parses cleanly: the channel notes are
        stripped by the parser (never re-printed) and the overlap guard cuts an
        echoed tail, so notes do not stack across rounds (the L2 round-trip
        law). Live values reach the writer through :attr:`scope`, not by
        interpolating them into this document.

        Rounds are joined by a BLANK LINE (each fragment already ends in one
        ``\\n``; the join adds the second) so a round boundary is a real
        markdown block break — two authored prose rounds never glue into one
        cell, and the concatenated document re-parses to the same cell
        structure."""
        return "\n".join(self._rendered_parts)

    @property
    def continued(self) -> str:
        """Alias for :attr:`rendered`, named for what it is used for.

        ``rendered`` is the document handed BACK to the writer to continue from,
        which is not what the name suggests to a caller looking for output. The
        alias makes the round-trip role explicit without changing what
        ``rendered`` means for existing callers.
        """
        return self.rendered

    @property
    def display(self) -> str:
        """The finished document a READER reads: executed output, all rounds.

        Prose with its interpolations expanded and whatever the cells printed —
        no code source, no ``[kernel]`` channel. This is ``clean_doc``
        accumulated across the session; ``clean_doc`` alone is one round's worth,
        which is the wrong artifact to show for a multi-round document.

        Never feed this back to the writer: the flat stdout re-prints and stacks
        when re-emitted (the exp1 poisoning). Use :attr:`continued` for that.
        """
        return "\n".join(p for p in self._clean_parts if p)

    @property
    def markdown(self) -> str:
        """The document as PORTABLE markdown: code in fences, no kernel channel.

        For publishing or viewing outside lackpy — a README, a docs page, a
        markdown viewer — where ```lackpy fences render and highlight but a
        ``<compute>`` tag is raw HTML. Source-preserving like :attr:`rendered`,
        not executed like :attr:`display`.

        A block whose body contains a fence keeps a longer outer fence, so the
        payload survives as valid CommonMark.
        """
        return to_markdown(self.rendered)

    @property
    def scope(self) -> dict[str, str]:
        """What is currently *live* in the kernel (name -> type: repr summary).

        The thin-client feeds this to the model between rounds instead of
        re-injecting variables as text -- the values are already live in the kernel.
        """
        return self._kernel.get_scope()
