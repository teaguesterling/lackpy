"""L2 annotation channel — the round-trip law's grammar surface.

The round-trip law (L2): *anything the kernel renders MUST be legal parser
input.* Kernel-generated notes therefore travel through a dedicated
**annotation channel** so that a rendered document, fed back to a stateless
writer and re-emitted, re-parses cleanly instead of re-printing and stacking.

The channel is a single construct with two surface forms::

    [kernel] warning: cell truncated [/kernel]      # inline, one line
    [kernel]                                         # block, multi-line
    ledger: 3 holes bound this round
    [/kernel]

Why ``[kernel]`` and not ``#`` (the brief's loose example): at the *document*
level the grammar is markdown, where ``#`` opens a heading and ``<!-- -->`` is
already meaningful. ``[kernel]`` reads to CommonMark as an undefined shortcut
reference — i.e. inert literal text that lands in a prose region — so a
line/span strip over the raw source is unambiguous and never collides with an
authored construct. Collapsing the brief's mechanisms (a) *comment syntax* and
(c) *[kernel] block* into one channel means a single grammar addition satisfies
both: text inside the delimiters is parsed as an inert annotation (a), and the
block form is ignored by binding parsers (c).

Two consumers:

* **Parsers** (batch ``parser.parse`` and streaming ``StreamingCellParser``)
  call :func:`strip_kernel_blocks` on prose regions, so channel spans never
  become prose cells — they are inert on reparse (mechanism a + c).
* **Renderers** (``formats.render_markdown``) call
  :func:`strip_kernel_annotations` on authored prose *before* re-emitting, then
  regenerate fresh notes through :func:`kernel_note` — no stale annotations
  accumulate across round-trips (mechanism b, "strip-stale-before-annotate").

**Scope discipline (document is sole source of truth).** The channel span
strip is safe to apply broadly: ``[kernel]…[/kernel]`` is a reserved,
kernel-only delimiter. The *legacy* bare-literal strip is deliberately narrow —
it matches ONLY the exact strings the kernel historically emitted before the
channel existed (the truncation warning and the ``@scratch`` summary), anchored
full-line. A user who writes "[warning: don't touch this]" in authored prose is
left untouched. The emitter literals are single-sourced here (see
:data:`TRUNCATION_NOTE`) so the emit path and the strip path cannot drift.
"""

from __future__ import annotations

import re

# --- Channel delimiters ----------------------------------------------------

KERNEL_OPEN = "[kernel]"
KERNEL_CLOSE = "[/kernel]"


def kernel_note(text: str) -> str:
    """Render ``text`` as an inline annotation through the channel.

    The result is legal parser input: parsers strip the span, so the note is
    inert on reparse (never re-printed as prose).
    """
    return f"{KERNEL_OPEN} {text} {KERNEL_CLOSE}"


# A [kernel] ... [/kernel] span, inline or spanning multiple lines. Non-greedy
# so adjacent spans don't merge; DOTALL so the block form is captured whole.
_KERNEL_SPAN_RE = re.compile(r"\[kernel\].*?\[/kernel\]", re.DOTALL)


# --- Legacy kernel-emitted bare literals (pre-channel) ---------------------
#
# These are the EXACT formats the kernel emitted before the annotation channel
# existed. They are what "poisoned" fed-back transcripts: a stateless writer
# imitated them, emitting them as bare prose that re-printed on the next round.
# Matching is scoped to the precise kernel formats, anchored full-line, so
# authored prose that merely resembles them is never eaten.

# Single source of truth: the streaming driver emits this to stdout as
# ``[warning: {TRUNCATION_NOTE}]`` (display path) and render_markdown emits it
# through the channel as ``[kernel] {TRUNCATION_NOTE} [/kernel]`` (canonical
# path). Import this in the driver so the emit literal and the strip regex below
# can never drift apart.
TRUNCATION_NOTE = "cell may be truncated — model hit token limit"

_LEGACY_TRUNCATION_LINE = re.compile(
    r"^[ \t]*\[warning: " + re.escape(TRUNCATION_NOTE) + r"\][ \t]*$"
)

# The @scratch summary format emitted by compiler.py ``_compile_scratch``:
#   [scratch: name=Type, name=Type]   (or [scratch: ] when NO new name bound —
#   the f-string always writes a space after the colon). Mirrors compiler.py:201;
#   keep the shape in sync if that codegen changes. The fixed space after the
#   colon is matched OUTSIDE the optional list so the empty form is covered.
_LEGACY_SCRATCH_LINE = re.compile(
    r"^[ \t]*\[scratch: (?:\w+=\w+(?:, \w+=\w+)*)?\][ \t]*$"
)


def strip_kernel_blocks(text: str) -> str:
    """Remove ``[kernel]…[/kernel]`` channel spans from ``text``.

    Used by the parsers so channel notes are inert — never re-parsed as prose.
    Only the reserved channel delimiter is touched; authored prose is preserved.
    """
    return _KERNEL_SPAN_RE.sub("", text)


def strip_kernel_annotations(text: str) -> str:
    """Strip-stale for renderers (mechanism b).

    Removes channel spans AND the exact legacy kernel-emitted bare-literal lines
    (truncation warning, @scratch summary), so re-annotating a document never
    stacks stale notes. Scoped strictly to kernel formats — authored content is
    left as written.
    """
    text = _KERNEL_SPAN_RE.sub("", text)
    kept: list[str] = []
    for line in text.split("\n"):
        if _LEGACY_TRUNCATION_LINE.match(line) or _LEGACY_SCRATCH_LINE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)
