"""LackpyToolkit — primary entry point for langchain-lackpy integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import PrivateAttr

from lackpy import LackpyService
from lackpy.config import load_config
from lackpy.kit.registry import ResolvedKit, resolve_kit

from ._delegate import LackpyDelegateTool
from ._tool_wrapper import LackpyToolWrapper


class LackpyToolkit(BaseToolkit):
    """Wraps a lackpy kit as a langchain BaseToolkit.

    All langchain-lackpy integration flows through this class.
    Construct it with a service and kit, then call get_tools(),
    as_delegate(), or as_node() to get langchain-compatible objects.
    """

    _service: Any = PrivateAttr()
    _kit_config: Any = PrivateAttr()
    _resolved_kit: ResolvedKit = PrivateAttr()

    def __init__(
        self,
        service: LackpyService,
        kit: str | list[str] | dict | None = None,
        kits_dir: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._service = service
        self._kit_config = kit
        self._resolved_kit = resolve_kit(kit, service.toolbox, kits_dir=kits_dir)

    @classmethod
    def from_config(
        cls,
        workspace: Path,
        kit: str | list[str] | dict | None = None,
        kits_dir: Path | None = None,
    ) -> LackpyToolkit:
        config = load_config(workspace)
        service = LackpyService(workspace=workspace, config=config)
        return cls(service=service, kit=kit, kits_dir=kits_dir)

    def get_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for alias, spec in self._resolved_kit.tools.items():
            callable_fn = self._resolved_kit.callables[alias]
            tools.append(LackpyToolWrapper.from_spec(spec, callable_fn, name_override=alias))
        return tools

    def as_delegate(
        self,
        name: str = "lackpy_delegate",
        description: str | None = None,
    ) -> BaseTool:
        return LackpyDelegateTool.create(
            service=self._service,
            kit_config=self._kit_config,
            resolved_description=self._resolved_kit.description,
            name=name,
            description=description,
        )

    def as_node(
        self,
        intent_key: str = "intent",
        result_key: str = "lackpy_result",
    ) -> Callable[..., Any]:
        from ._node import make_node
        return make_node(
            service=self._service,
            kit_config=self._kit_config,
            intent_key=intent_key,
            result_key=result_key,
        )
