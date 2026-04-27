"""Tests for ArgSpec type string → Python type mapping."""

from __future__ import annotations

import pytest
from typing import Any

from lackpy.kit.toolbox import ARGSPEC_TYPE_MAP, resolve_python_type


class TestArgspecTypeMap:
    def test_str_maps_to_str(self):
        assert resolve_python_type("str") is str

    def test_int_maps_to_int(self):
        assert resolve_python_type("int") is int

    def test_float_maps_to_float(self):
        assert resolve_python_type("float") is float

    def test_bool_maps_to_bool(self):
        assert resolve_python_type("bool") is bool

    def test_dict_maps_to_dict(self):
        assert resolve_python_type("dict") is dict

    def test_list_maps_to_list(self):
        assert resolve_python_type("list") is list

    def test_any_maps_to_any(self):
        assert resolve_python_type("Any") is Any

    def test_unknown_falls_back_to_any(self):
        assert resolve_python_type("SomeCustomType") is Any

    def test_list_str_falls_back_to_any(self):
        assert resolve_python_type("list[str]") is Any

    def test_map_contains_all_base_types(self):
        expected = {"str", "int", "float", "bool", "dict", "list", "Any"}
        assert set(ARGSPEC_TYPE_MAP.keys()) == expected
