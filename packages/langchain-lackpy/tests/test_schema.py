"""Tests for ArgSpec → Pydantic args_schema conversion."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from lackpy.kit.toolbox import ArgSpec

from langchain_lackpy._schema import args_schema_from_argspecs


class TestArgsSchemaFromArgspecs:
    def test_single_str_arg(self):
        specs = [ArgSpec(name="path", type="str", description="File path")]
        model = args_schema_from_argspecs("read_file", specs)
        assert issubclass(model, BaseModel)
        fields = model.model_fields
        assert "path" in fields
        assert fields["path"].annotation is str

    def test_multiple_args(self):
        specs = [
            ArgSpec(name="path", type="str", description="File path"),
            ArgSpec(name="content", type="str", description="Content"),
        ]
        model = args_schema_from_argspecs("write_file", specs)
        assert set(model.model_fields.keys()) == {"path", "content"}

    def test_int_arg_type(self):
        specs = [ArgSpec(name="count", type="int", description="Number of items")]
        model = args_schema_from_argspecs("my_tool", specs)
        assert model.model_fields["count"].annotation is int

    def test_any_arg_type(self):
        from typing import Any
        specs = [ArgSpec(name="data", type="Any", description="Arbitrary data")]
        model = args_schema_from_argspecs("my_tool", specs)
        assert model.model_fields["data"].annotation is Any

    def test_unknown_type_falls_back_to_any(self):
        from typing import Any
        specs = [ArgSpec(name="data", type="SomeCustomType", description="Custom")]
        model = args_schema_from_argspecs("my_tool", specs)
        assert model.model_fields["data"].annotation is Any

    def test_field_descriptions_set(self):
        specs = [ArgSpec(name="path", type="str", description="The file path")]
        model = args_schema_from_argspecs("read_file", specs)
        assert model.model_fields["path"].description == "The file path"

    def test_model_name_derived_from_tool(self):
        specs = [ArgSpec(name="x", type="str")]
        model = args_schema_from_argspecs("read_file", specs)
        assert model.__name__ == "ReadFileInput"

    def test_empty_args_produces_empty_model(self):
        model = args_schema_from_argspecs("no_args_tool", [])
        assert len(model.model_fields) == 0

    def test_model_validates_input(self):
        specs = [ArgSpec(name="count", type="int", description="N")]
        model = args_schema_from_argspecs("my_tool", specs)
        instance = model(count=5)
        assert instance.count == 5
