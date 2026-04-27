"""LangGraph node factory for lackpy delegate."""

from __future__ import annotations

from typing import Any, Callable


def make_node(
    service: Any,
    kit_config: Any,
    intent_key: str = "intent",
    result_key: str = "lackpy_result",
) -> Callable[..., Any]:
    """Return an async function shaped for StateGraph.add_node().

    The returned function reads ``state[intent_key]``, calls
    ``service.delegate()``, and returns ``{result_key: result}``.
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        intent = state[intent_key]
        result = await service.delegate(intent=intent, kit=kit_config)
        return {result_key: result}

    return node
