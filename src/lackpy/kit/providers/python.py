"""Python import-based tool provider."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from ..toolbox import ToolSpec


class PythonProvider:
    @property
    def name(self) -> str:
        return "python"

    def available(self) -> bool:
        return True

    def resolve(self, tool_spec: ToolSpec) -> Callable[..., Any]:
        module_name = tool_spec.provider_config.get("module")
        function_name = tool_spec.provider_config.get("function")
        if not module_name or not function_name:
            raise ValueError(
                f"Python provider for '{tool_spec.name}' requires "
                f"'module' and 'function' in provider_config"
            )
        mod = importlib.import_module(module_name)
        fn = getattr(mod, function_name, None)
        if fn is None:
            raise AttributeError(f"Module '{module_name}' has no function '{function_name}'")
        return fn
