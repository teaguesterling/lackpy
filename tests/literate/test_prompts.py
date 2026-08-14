"""Tests for the persona + interpreter prompt composition system."""

import pytest

from lackpy.prompts import PERSONAS, compose, list_personas
from lackpy.interpreters.literate import LiterateInterpreter


class TestCompose:
    def test_all_personas_compose_with_literate(self):
        interp = LiterateInterpreter()
        for name in list_personas():
            result = compose(name, interp)
            assert "{interpreter_hint}" not in result
            assert "<compute>" in result
            assert "read_file" in result

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
        """Composed prompts should stay under 6000 chars for smaller models."""
        interp = LiterateInterpreter()
        for name in list_personas():
            result = compose(name, interp)
            # Raised from 6000 when <compute> became the documented syntax: the
            # tag form costs ~320 chars over the fence form (attributes are
            # wordier than @annotations, and every block gains a closing tag),
            # and the fence-era prompt already sat at 5931. The guard against
            # prompt bloat is kept, just re-based on the new floor.
            assert len(result) < 6400, (
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
        """The shipped conventions teach bind-through-the-unknown: an unknown
        name binds a hole and forward references are legal."""
        hint = self._hint()
        assert "hole" in hint
        assert "⟨name: unbound⟩" in hint  # the merged Hole repr
        assert "forward" in hint  # forward references are now legal

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
