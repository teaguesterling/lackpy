"""Tests for doc refinement callbacks and correction chain doc integration."""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from lackpy.infer.correction import _infer_relevant_tool
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


class TestInferRelevantTool:
    def test_open_maps_to_read_file(self):
        result = _infer_relevant_tool(
            "stdlib_leak",
            ["Forbidden name: 'open' at line 1"],
            {"read_file", "find_files"},
        )
        assert result == "read_file"

    def test_glob_maps_to_find_files(self):
        result = _infer_relevant_tool(
            "stdlib_leak",
            ["Forbidden name: 'glob' at line 2"],
            {"read_file", "find_files"},
        )
        assert result == "find_files"

    def test_os_maps_to_find_files(self):
        result = _infer_relevant_tool(
            "implement_not_orchestrate",
            ["Forbidden name: 'os' at line 1"],
            {"find_files", "edit_file"},
        )
        assert result == "find_files"

    def test_pathlib_maps_to_find_files(self):
        result = _infer_relevant_tool(
            "stdlib_leak",
            ["Forbidden name: 'pathlib' at line 1"],
            {"read_file", "find_files"},
        )
        assert result == "find_files"

    def test_no_match_falls_back_to_underscore_name(self):
        result = _infer_relevant_tool(
            "stdlib_leak",
            ["Forbidden name: 'subprocess' at line 1"],
            {"read_file", "find_files"},
        )
        assert result in ("read_file", "find_files")

    def test_no_underscore_names_returns_none(self):
        result = _infer_relevant_tool(
            "stdlib_leak",
            ["Forbidden name: 'subprocess'"],
            {"len", "sorted"},
        )
        assert result is None

    def test_unrelated_failure_mode_uses_fallback(self):
        result = _infer_relevant_tool(
            "path_prefix",
            ["some error"],
            {"read_file"},
        )
        assert result == "read_file"

    def test_none_failure_mode(self):
        result = _infer_relevant_tool(
            None,
            ["some error"],
            {"read_file"},
        )
        assert result == "read_file"

    def test_empty_errors_and_names(self):
        result = _infer_relevant_tool("stdlib_leak", [], set())
        assert result is None

    def test_tool_not_in_allowed_names_skipped(self):
        result = _infer_relevant_tool(
            "stdlib_leak",
            ["Forbidden name: 'open'"],
            {"find_files"},  # read_file not in allowed
        )
        assert result == "find_files"  # fallback to underscore name
