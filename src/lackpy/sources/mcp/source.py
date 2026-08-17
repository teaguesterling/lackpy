"""McpToolSource — a ToolSource backed by one MCP server.

Discovers the server's tools (via a shared :class:`McpClient`), builds full
``ToolSpec``s (inputSchema → args, annotations → grade), and resolves callables
to async proxies that marshal ``call_tool`` onto the client loop.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from ...tools.toolbox import ArgSpec, ToolSpec
from ...run.bridge import mark_async
from .client import McpClient, McpServerSpec
from .grade import grade_from_annotations

# JSON Schema primitive → lackpy ArgSpec type string (display + resolve_python_type).
_JSON_TYPE_MAP = {
    "string": "str", "integer": "int", "number": "float",
    "boolean": "bool", "object": "dict", "array": "list",
}


class McpToolSource:
    """Exposes one MCP server's tools as lackpy tools.

    Args:
        spec: Connection spec for the server.
        client: Shared MCP client owning the background loop/sessions.
        grade_overrides: Optional ``{tool_name: (w, d)}`` overrides (config wins).
        example_overrides: Optional ``{tool_name: [{"intent": ..., "code": ...}]}``
            few-shot examples, from ``[mcp_servers.<id>.tools.<name>] examples``.
            Discovery gives an MCP tool a name, a description and an argument
            schema, but nothing about *usage* — so a generator sees the signature
            and has to guess the idiom. Examples reach the prompt through the same
            retrieval path as builtin tools' (``collect_example_pool`` ->
            ``retrieve_examples``), which is a different channel from prose in the
            intent and is not pattern-completable the way an example value in
            prose is.
        connect_timeout: Per-server connect/discovery timeout (seconds).
    """

    def __init__(self, spec: McpServerSpec, client: McpClient,
                 grade_overrides: dict[str, tuple[int, int]] | None = None,
                 example_overrides: dict[str, list[dict]] | None = None,
                 connect_timeout: float = 30.0) -> None:
        self._spec = spec
        self._client = client
        self._grade_overrides = grade_overrides or {}
        self._example_overrides = example_overrides or {}
        self._connect_timeout = connect_timeout
        self._discovered: list[Any] | None = None

    @property
    def name(self) -> str:
        return f"mcp:{self._spec.server_id}"

    def available(self) -> bool:
        try:
            self._ensure_discovered()
            return True
        except Exception:
            return False

    def discover(self) -> list[ToolSpec]:
        return [self._to_spec(t) for t in self._ensure_discovered()]

    def resolve(self, spec: ToolSpec) -> Callable[..., Any]:
        server_id = self._spec.server_id
        client = self._client
        mcp_name = spec.provider_config.get("mcp_name", spec.name)
        param_names = [a.name for a in spec.args]

        def proxy(*args: Any, **kwargs: Any) -> Any:
            call_args = dict(zip(param_names, args))
            call_args.update(kwargs)
            return client.call(server_id, mcp_name, call_args)

        # A real signature so make_traced maps positional args to names.
        proxy.__signature__ = inspect.Signature(
            [inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in param_names]
        )
        proxy.__name__ = spec.name
        return mark_async(proxy)

    # ---------- internals ----------

    def _ensure_discovered(self) -> list[Any]:
        if self._discovered is None:
            self._discovered = self._client.connect(self._spec, timeout=self._connect_timeout)
        return self._discovered

    def _to_spec(self, tool: Any) -> ToolSpec:
        override = self._grade_overrides.get(tool.name)
        if override is not None:
            gw, gd = override
        else:
            g = grade_from_annotations(getattr(tool, "annotations", None))
            gw, gd = g.w, g.d
        return ToolSpec(
            name=tool.name,
            provider=self.name,
            provider_config={"mcp_name": tool.name},
            description=tool.description or "",
            args=_args_from_schema(getattr(tool, "inputSchema", None)),
            returns=returns_from_schema(getattr(tool, "outputSchema", None)),
            examples=self._example_overrides.get(tool.name, []),
            grade_w=gw,
            effects_ceiling=gd,
        )


def _args_from_schema(schema: Any) -> list[ArgSpec]:
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties", {})
    args: list[ArgSpec] = []
    for pname, pdef in props.items():
        pdef = pdef if isinstance(pdef, dict) else {}
        json_type = pdef.get("type", "Any")
        args.append(ArgSpec(
            name=pname,
            type=_JSON_TYPE_MAP.get(json_type, "Any"),
            description=pdef.get("description", ""),
        ))
    return args


def returns_from_schema(schema: Any) -> str:
    """Render an MCP ``outputSchema`` as a lackpy ``returns`` annotation.

    ``Toolbox.format_description`` renders each tool as
    ``name(args) -> returns: description``, so ``returns`` is where a shape
    belongs — beside the signature rather than competing with it as prose. Every
    MCP tool previously arrived as ``"Any"``, leaving the generator to guess what
    a call handed back; a wrong guess yields a program that validates, runs and
    answers incorrectly, which the AST whitelist cannot catch.

    Unwraps fastmcp's ``{"properties": {"result": ...}}`` envelope, then maps
    JSON-Schema types onto the Python-ish names the prompt already uses. Falls
    back to ``"Any"`` so an absent or unrecognised schema behaves as before.

    Note the ceiling: fastmcp emits ``{"type": "object", "additionalProperties":
    true}`` for any dict-returning tool without a model annotation, which carries
    no key names. Such a tool improves only from ``Any`` to ``dict`` — knowing a
    value is a dict does not tell you which key holds the answer.
    """
    if not isinstance(schema, dict):
        return "Any"
    props = schema.get("properties")
    if isinstance(props, dict) and set(props) == {"result"}:
        return returns_from_schema(props["result"])
    if isinstance(schema.get("anyOf"), list):
        parts = [returns_from_schema(x) for x in schema["anyOf"]]
        parts = [p for p in dict.fromkeys(parts) if p != "Any"]
        return " | ".join(parts) if parts else "Any"
    t = schema.get("type")
    if t in ("string", "integer", "number", "boolean"):
        return {"string": "str", "integer": "int",
                "number": "float", "boolean": "bool"}[t]
    if t == "array":
        return f"list[{returns_from_schema(schema.get('items'))}]"
    if t == "object":
        keys = list(schema.get("properties") or {})
        if keys:
            return "dict{" + ", ".join(keys[:6]) + ("…}" if len(keys) > 6 else "}")
        return "dict"
    return "Any"
