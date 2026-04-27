"""Individual ToolSpec → BaseTool wrapper."""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool
from pydantic import BaseModel, PrivateAttr

from lackpy.kit.toolbox import ToolSpec

from ._schema import args_schema_from_argspecs


class LackpyToolWrapper(BaseTool):
    """A langchain BaseTool wrapping a single lackpy ToolSpec and its callable."""

    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] = BaseModel
    metadata: dict[str, Any] = {}
    _callable: Callable[..., Any] = PrivateAttr()

    @classmethod
    def from_spec(
        cls,
        spec: ToolSpec,
        callable_fn: Callable[..., Any],
        name_override: str | None = None,
    ) -> LackpyToolWrapper:
        tool_name = name_override or spec.name
        schema = args_schema_from_argspecs(tool_name, spec.args)
        description = f"{spec.description} [provider={spec.provider}, grade_w={spec.grade_w}]"
        instance = cls(
            name=tool_name,
            description=description,
            args_schema=schema,
            metadata={
                "provider": spec.provider,
                "grade_w": spec.grade_w,
                "effects_ceiling": spec.effects_ceiling,
            },
        )
        instance._callable = callable_fn
        return instance

    def _run(self, **kwargs: Any) -> str:
        result = self._callable(**kwargs)
        return str(result)
