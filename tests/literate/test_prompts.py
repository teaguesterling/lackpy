"""Tests for the persona + interpreter prompt composition system."""

import pytest

from lackpy.lang.spec import format_preprompt

from lackpy.prompts import PERSONAS, compose, list_personas
from lackpy.interpreters.literate import LiterateInterpreter


class TestCompose:
    def test_all_personas_compose_with_literate(self):
        interp = LiterateInterpreter()
        for name in list_personas():
            result = compose(name, interp)
            assert "{interpreter_hint}" not in result
            assert "<compute>" in result
            # Literate stepy Task 5 (retired kit model: asserted "read_file"):
            # the prompt teaches Path, not tool calls.
            assert "read_file" not in result
            assert "Path(" in result

    def test_compose_with_raw_string(self):
        result = compose("general", "Output JSON only.")
        assert "Output JSON only." in result
        assert "{interpreter_hint}" not in result

    def test_unknown_persona_raises(self):
        with pytest.raises(ValueError, match="Unknown persona"):
            compose("nonexistent", "hint")

    def test_invalid_interpreter_type_raises(self):
        with pytest.raises(TypeError, match="system_prompt_hint"):
            compose("general", 42)

    def test_personas_registry(self):
        names = list_personas()
        assert "general" in names
        assert "analyst" in names
        assert "developer" in names

    def test_each_persona_has_format_slot(self):
        for name, template in PERSONAS.items():
            assert "{interpreter_hint}" in template, (
                f"Persona {name!r} missing {{interpreter_hint}} slot"
            )

    def test_literate_hint_contains_key_syntax(self):
        hint = LiterateInterpreter().system_prompt_hint()
        assert "<compute hidden>" in hint
        assert "<compute gather>" in hint
        assert "<compute continue>" in hint
        assert '<compute write="path">' in hint
        assert '<compute diff="path">' in hint
        assert '<compute read="path">' in hint
        assert "<compute scratch>" in hint

    def test_composed_prompt_not_too_long(self):
        """Composed prompts stay bounded: the lesson, the format, one document."""
        interp = LiterateInterpreter()
        for name in list_personas():
            result = compose(name, interp)
            # Raised from 6000 when <compute> became the documented syntax: the
            # tag form costs ~320 chars over the fence form (attributes are
            # wordier than @annotations, and every block gains a closing tag),
            # and the fence-era prompt already sat at 5931. The guard against
            # prompt bloat is kept, just re-based on the new floor.
            # Task 5: re-based on the stepy lesson (format_preprompt, ~3k) plus
            # the worked document. The tool list (~400) is gone.
            assert len(result) < 11000, (
                f"Persona {name!r} composed to {len(result)} chars"
            )


class TestL5ForgivenessConventions:
    """L5: surface conventions ship WITH the kernel and advertise forgiveness.

    The assertions anchor on STABLE tokens (``hole``, ``⟨…⟩``, ``forward``,
    ``@continue``, ``manifest``/``budget``, ``[kernel]``), never on the exact
    provisional sentence of the bind-through-holes clause — that wording is a
    PARKED open question (L5 decay flag) Teague will swap after his taught-arm
    ablation. Anchoring on tokens keeps the wording tunable and keeps these
    tests from encoding (i.e. resolving) the parked question.
    """

    def _hint(self) -> str:
        return LiterateInterpreter().system_prompt_hint()

    def test_conventions_teach_bind_through_holes(self):
        """Task 5 fix round (retired: 'forward references are legal'): the hint
        teaches define-before-use; a hole is named only as recovery."""
        hint = " ".join(self._hint().split())
        assert "hole" in hint
        assert "⟨name: unbound⟩" in hint  # the merged Hole repr
        assert "forward references are legal" not in hint
        assert "may reference results before you define them" not in hint
        assert "Bind through the unknown" not in hint
        assert "Define every name before you use it" in hint
        assert "recovery" in hint

    def test_old_define_before_using_hint_removed(self):
        """The REVERSAL: the old 'define before using / no forward references'
        guidance (and the streaming-path 'patch-forward' framing) is GONE."""
        hint = self._hint()
        assert "No forward references" not in hint
        assert "define variables BEFORE" not in hint
        assert "patch-forward" not in hint
        assert "caught before execution" not in hint

    def test_conventions_cover_four_affordances(self):
        """All four L5 affordances are present in the shipped conventions."""
        hint = self._hint()
        # 1. bind-through-the-unknown
        assert "hole" in hint
        # 2. kernel arithmetic authority (kernel computes/evaluates; the
        #    reserved [kernel] channel is the kernel's, not the writer's)
        assert "kernel computes and evaluates" in hint
        assert "[kernel]" in hint
        # 3. pause protocol
        assert "<compute continue>" in hint
        # 4. visible budget
        assert "manifest" in hint
        assert "budget" in hint

    def test_conventions_shipped_with_kernel(self):
        """The conventions are reachable from the KERNEL's prompt surface —
        the interpreter's own hint and every composed persona — not just a
        standalone doc file."""
        # Reachable from the interpreter (the kernel deliverable's surface).
        hint = self._hint()
        assert "⟨name: unbound⟩" in hint
        # And they ship through the composition surface every persona uses.
        interp = LiterateInterpreter()
        for name in list_personas():
            composed = compose(name, interp)
            assert "hole" in composed
            assert "<compute continue>" in composed
            assert "manifest" in composed


# ---------------------------------------------------------------------------
# Literate stepy Task 5 (D10): generation teaches the stepy language.
# ---------------------------------------------------------------------------

_BANNED = ["read_file", "find_files", "write_file", "edit_file", "apply_diff",
           "search_content", "run_command", "run_tests", "subprocess",
           "import os", "os.path"]


def _every_literate_prompt() -> dict[str, str]:
    from lackpy.infer.prompt import build_system_prompt
    from lackpy.interpreters.literate.prompt import LITERATE_HINT, LITERATE_SYSTEM_PROMPT

    interp = LiterateInterpreter()
    prompts = {
        "hint": LITERATE_HINT,
        "system": LITERATE_SYSTEM_PROMPT,
        "build_system_prompt": build_system_prompt(namespace_desc="", interpreter=interp),
    }
    for name in list_personas():
        prompts[f"persona:{name}"] = compose(name, interp)
    return prompts


def test_no_kit_or_subprocess_names_in_literate_prompt():
    for where, prompt in _every_literate_prompt().items():
        for word in _BANNED:
            assert word not in prompt, f"{word!r} in {where}"


def test_literate_prompt_is_the_lesson_then_the_format_then_a_document():
    from lackpy.interpreters.literate.prompt import LITERATE_HINT

    assert LITERATE_HINT.startswith(format_preprompt())
    lesson = LITERATE_HINT.index(format_preprompt())
    fmt = LITERATE_HINT.index("## Annotations")
    doc = LITERATE_HINT.index("<compute hidden>", fmt + 200)
    assert lesson < fmt < doc


def test_literate_prompt_teaches_the_ruled_points():
    from lackpy.interpreters.literate.prompt import LITERATE_HINT

    flat = " ".join(LITERATE_HINT.split())
    # Annotation cells import Path themselves: the document must not import it again.
    assert "do not import `Path` again" in flat
    # Output is the trailing expression of the last code cell.
    assert "last code cell" in flat and "trailing expression" in flat
    # @diff is applied by lackpy and has no python3 twin.
    assert "applied by lackpy" in flat and "python3" in flat


def _worked_document() -> str:
    from lackpy.interpreters.literate.prompt import LITERATE_HINT

    start = "=== EXAMPLE DOCUMENT ===\n"
    end = "\n=== END EXAMPLE DOCUMENT ==="
    assert start in LITERATE_HINT and end in LITERATE_HINT
    return LITERATE_HINT.split(start, 1)[1].split(end, 1)[0] + "\n"


@pytest.mark.asyncio
async def test_the_worked_document_in_the_prompt_runs(tmp_path):
    """The one document the prompt shows must run through the service's literate
    path, in a tmp workspace holding the file it reads (controller decision)."""
    from lackpy.profiles import Profile
    from lackpy.run.paths import load_policy
    from lackpy.service import LackpyService

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "scores.txt").write_text("Alice: 90\nBob: 85\nCarol: 92\n")
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    svc = LackpyService(
        workspace=ws,
        path_policy_loader=lambda w: load_policy(w, home=home, managed=None),
    )
    doc = _worked_document()
    result = await svc.run_program(doc, profile=Profile(execution="literate"))
    assert result.success, result.error
    assert "Carol" in result.stdout
    # The prompt's output rule: the last code cell's trailing expression.
    assert result.output == "Carol"
    # I2: the file holds the computed winner, not a fixed sentence.
    assert (ws / "best.txt").read_text() == "Carol"


def test_literate_prompt_says_write_is_literal_and_a_refusal_ends_the_document():
    from lackpy.interpreters.literate.prompt import LITERATE_HINT

    flat = " ".join(LITERATE_HINT.split())
    assert "literal file content only" in flat
    assert "A refusal ends the document" in flat
    assert "line that the full style guide follows: it does not" in flat
    assert "key=scores.get" not in flat
