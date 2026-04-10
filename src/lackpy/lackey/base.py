"""Lackey base class — the foundation for lackpy program modules."""

from __future__ import annotations

from typing import Any

from .tool import Tool

_RESERVED = frozenset({
    "returns", "creation_log",
    "run", "get_tool_names", "get_param_specs", "get_returns",
    "_tool_descriptors", "_param_specs", "_resolved_tools",
})


def _get_class_annotations(cls: type, namespace: dict[str, Any]) -> dict[str, Any]:
    """Return a class's own annotations, compatible with Python 3.10-3.14+.

    Python 3.14 (PEP 649) made annotations lazy — they live in
    ``__annotate_func__`` rather than a direct ``__annotations__`` dict in
    the class namespace. ``inspect.get_annotations(cls)`` handles both
    storage forms and only returns the class's own annotations (not
    inherited ones).
    """
    import inspect
    try:
        return inspect.get_annotations(cls)
    except Exception:
        # Fallback for edge cases: pre-3.10 used namespace.get directly
        return namespace.get("__annotations__", {})


class LackeyMeta(type):
    def __new__(mcs, name: str, bases: tuple, namespace: dict[str, Any]) -> type:
        cls = super().__new__(mcs, name, bases, namespace)
        if name == "Lackey":
            return cls

        tool_descriptors: dict[str, Tool] = {}
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, Tool):
                tool_descriptors[attr_name] = attr_value
        cls._tool_descriptors = tool_descriptors

        param_specs: dict[str, dict[str, Any]] = {}
        annotations = _get_class_annotations(cls, namespace)
        for param_name, param_type in annotations.items():
            if param_name in _RESERVED or param_name in tool_descriptors or param_name.startswith("_"):
                continue
            spec: dict[str, Any] = {"type": param_type}
            if param_name in namespace:
                spec["default"] = namespace[param_name]
            param_specs[param_name] = spec
        cls._param_specs = param_specs

        return cls


class Lackey(metaclass=LackeyMeta):
    """Base class for lackpy programs."""

    returns: Any = None
    creation_log: Any = None

    _tool_descriptors: dict[str, Tool] = {}
    _param_specs: dict[str, dict[str, Any]] = {}
    _resolved_tools: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        for name, spec in self._param_specs.items():
            if name in kwargs:
                setattr(self, name, kwargs.pop(name))
            elif "default" in spec:
                setattr(self, name, spec["default"])

        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown parameter(s): {unknown}")

        self._resolved_tools = {}

    def run(self) -> Any:
        raise NotImplementedError("Subclasses must implement run()")

    @classmethod
    def get_tool_names(cls) -> list[str]:
        return list(cls._tool_descriptors.keys())

    @classmethod
    def get_param_specs(cls) -> dict[str, dict[str, Any]]:
        return dict(cls._param_specs)

    @classmethod
    def get_returns(cls) -> Any:
        returns = cls.__dict__.get("returns")
        if returns is not None:
            return returns
        run_method = cls.__dict__.get("run")
        if run_method and hasattr(run_method, "__annotations__"):
            return run_method.__annotations__.get("return")
        return None
