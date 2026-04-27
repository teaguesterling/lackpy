"""Delegate BaseTool wrapping LackpyService.delegate()."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, PrivateAttr


class DelegateInput(BaseModel):
    intent: str = Field(description="Natural language description of what to accomplish")


class LackpyDelegateTool(BaseTool):
    """A langchain tool that delegates to lackpy's generate-then-execute pipeline."""

    name: str = "lackpy_delegate"
    description: str = ""
    args_schema: type[BaseModel] = DelegateInput
    _service: Any = PrivateAttr()
    _kit_config: Any = PrivateAttr()

    @classmethod
    def create(
        cls,
        service: Any,
        kit_config: Any,
        resolved_description: str,
        name: str = "lackpy_delegate",
        description: str | None = None,
    ) -> LackpyDelegateTool:
        if description is None:
            description = (
                "Safely compose operations into a validated, restricted Python program. "
                "Programs are graded for safety before execution.\n"
                f"Available operations:\n{resolved_description}"
            )
        instance = cls(name=name, description=description)
        instance._service = service
        instance._kit_config = kit_config
        return instance

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("LackpyDelegateTool is async-only. Use ainvoke().")

    async def _arun(self, intent: str, **kwargs: Any) -> str:
        try:
            result = await self._service.delegate(intent=intent, kit=self._kit_config)
        except Exception as exc:
            raise ToolException(str(exc)) from exc

        if result.get("success"):
            output = result.get("output")
            return str(output) if output is not None else ""
        return result.get("error") or "Unknown error"
