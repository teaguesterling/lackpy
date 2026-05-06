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
            assert "```lackpy" in result
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
        assert "@hidden" in hint
        assert "@gather" in hint
        assert "@continue" in hint
        assert "@write" in hint
        assert "@diff" in hint
        assert "@read" in hint
        assert "@scratch" in hint

    def test_composed_prompt_not_too_long(self):
        """Composed prompts should stay under 6000 chars for smaller models."""
        interp = LiterateInterpreter()
        for name in list_personas():
            result = compose(name, interp)
            assert len(result) < 6000, (
                f"Persona {name!r} composed to {len(result)} chars"
            )
