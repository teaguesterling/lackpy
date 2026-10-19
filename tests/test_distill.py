"""Tests for doc refinement callbacks and correction chain doc integration."""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from lackpy.infer.distill import _select_sections, _FAILURE_RELEVANT_SECTIONS


@dataclass
class FakeSection:
    title: str
    content: str = ""


class TestSelectSections:
    def test_stdlib_leak_selects_signature_params_notes(self):
        sections = [
            FakeSection("Signature"), FakeSection("Parameters"),
            FakeSection("Returns"), FakeSection("Notes"),
            FakeSection("Examples"),
        ]
        result = _select_sections(sections, {"failure_mode": "stdlib_leak"})
        titles = [s.title for s in result]
        assert titles == ["Signature", "Parameters", "Notes"]

    def test_implement_not_orchestrate_selects_signature_examples(self):
        sections = [
            FakeSection("Signature"), FakeSection("Parameters"),
            FakeSection("Returns"), FakeSection("Notes"),
            FakeSection("Examples"),
        ]
        result = _select_sections(sections, {"failure_mode": "implement_not_orchestrate"})
        titles = [s.title for s in result]
        assert titles == ["Signature", "Examples"]

    def test_key_hallucination_selects_signature_returns_notes(self):
        sections = [
            FakeSection("Signature"), FakeSection("Parameters"),
            FakeSection("Returns"), FakeSection("Notes"),
        ]
        result = _select_sections(sections, {"failure_mode": "key_hallucination"})
        titles = [s.title for s in result]
        assert titles == ["Signature", "Returns", "Notes"]

    def test_unknown_failure_mode_returns_first_two(self):
        sections = [
            FakeSection("Signature"), FakeSection("Parameters"),
            FakeSection("Returns"),
        ]
        result = _select_sections(sections, {"failure_mode": "something_new"})
        assert len(result) == 2
        assert result[0].title == "Signature"

    def test_no_matching_titles_returns_first_two(self):
        sections = [FakeSection("Overview"), FakeSection("History")]
        result = _select_sections(sections, {"failure_mode": "stdlib_leak"})
        assert len(result) == 2

    def test_empty_sections_returns_empty(self):
        result = _select_sections([], {"failure_mode": "stdlib_leak"})
        assert result == []

    def test_missing_failure_mode_key(self):
        sections = [FakeSection("Signature"), FakeSection("Notes")]
        result = _select_sections(sections, {})
        assert len(result) == 2

    def test_all_known_failure_modes_have_mappings(self):
        from lackpy.infer.failure_modes import ALL_MODES
        unmapped = ALL_MODES - set(_FAILURE_RELEVANT_SECTIONS.keys())
        # These modes don't have doc-based fixes
        expected_unmapped = {"jupyter_confusion", "syntax_artifact"}
        assert unmapped == expected_unmapped
