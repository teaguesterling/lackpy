"""ArgSpec list → Pydantic BaseModel conversion."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from pydantic import create_model as _create_model

from lackpy.kit.toolbox import ArgSpec, resolve_python_type


def args_schema_from_argspecs(tool_name: str, argspecs: list[ArgSpec]) -> type[BaseModel]:
    """Build a Pydantic model from a list of ArgSpecs.

    The model name is derived from the tool name: ``read_file`` becomes
    ``ReadFileInput``.
    """
    model_name = "".join(part.capitalize() for part in tool_name.split("_")) + "Input"

    field_definitions: dict[str, Any] = {}
    for arg in argspecs:
        python_type = resolve_python_type(arg.type)
        field_definitions[arg.name] = (
            python_type,
            Field(description=arg.description or None),
        )

    return _create_model(model_name, **field_definitions)
