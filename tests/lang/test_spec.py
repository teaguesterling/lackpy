"""Tests for the language spec module."""

from lackpy.lang.spec import get_spec


def test_get_spec_returns_dict():
    spec = get_spec()
    assert isinstance(spec, dict)


def test_get_spec_has_required_keys():
    spec = get_spec()
    assert "allowed_nodes" in spec
    assert "forbidden_nodes" in spec
    assert "forbidden_names" in spec
    assert "allowed_builtins" in spec


def test_get_spec_values_are_lists():
    spec = get_spec()
    assert isinstance(spec["allowed_nodes"], list)
    assert isinstance(spec["forbidden_nodes"], list)
    assert isinstance(spec["forbidden_names"], list)
    assert isinstance(spec["allowed_builtins"], list)


def test_get_spec_contains_known_entries():
    spec = get_spec()
    assert "Module" in spec["allowed_nodes"]
    assert "Import" in spec["forbidden_nodes"]
    assert "__import__" in spec["forbidden_names"]
    assert "len" in spec["allowed_builtins"]
