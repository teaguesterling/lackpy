"""Tests for output sanitization."""

from lackpy.infer.sanitize import sanitize_output


def test_strips_whitespace():
    assert sanitize_output("  x = 1  ") == "x = 1"


def test_strips_python_fence():
    assert sanitize_output("```python\nx = 1\n```") == "x = 1"


def test_strips_plain_fence():
    assert sanitize_output("```\nx = 1\n```") == "x = 1"


def test_strips_preamble():
    assert sanitize_output("Here's the code:\nx = 1") == "x = 1"


def test_strips_preamble_with_fence():
    assert sanitize_output("Here is the solution:\n```python\nx = 1\n```") == "x = 1"


def test_passthrough_clean_code():
    raw = "x = read('test.py')\nlen(x)"
    assert sanitize_output(raw) == raw


def test_handles_empty_string():
    assert sanitize_output("") == ""


def test_handles_only_fences():
    assert sanitize_output("```python\n```") == ""
