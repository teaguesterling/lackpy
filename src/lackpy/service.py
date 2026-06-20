"""Unified service layer: the API that MCP and CLI both call."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from .config import LackpyConfig, load_config
from .infer.dispatch import InferenceDispatcher, GenerationResult
from .infer.prompt import format_params_description
from .infer.providers.rules import RulesProvider
from .infer.providers.templates import TemplatesProvider
from .kit.providers.builtin import BuiltinProvider
from .kit.providers.python import PythonProvider
from .kit.registry import ResolvedKit
from .profiles import Profile, ResolvedProfile, resolve_profile
from .kit.toolbox import Toolbox
from .lang.grammar import ALLOWED_BUILTINS
from .policy.layer import PolicyLayer
from .policy.sources.kit import KitPolicySource
from .lang.validator import ValidationResult, validate
from .run.base import ExecutionResult
from .run.bridge import AsyncBridge, is_async_callable
from .run.runner import RestrictedRunner

logger = logging.getLogger(__name__)


def _strip_top_level_return(program: str) -> str:
    """Rewrite top-level ``return X`` statements to bare expression ``X``.

    Lackey run() bodies are extracted verbatim and may contain ``return``
    statements.  The lackpy runner executes programs as flat scripts (not
    inside a function), so bare ``return`` is a syntax error.  This helper
    converts each top-level ``return X`` to an expression statement ``X``
    so the runner can capture the value via its last-expression mechanism.
    """
    import ast
    tree = ast.parse(program)
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.Return):
            if node.value is not None:
                new_body.append(ast.Expr(
                    value=node.value,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ))
            # bare ``return`` (None) is simply dropped
        else:
            new_body.append(node)
    tree.body = new_body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


try:
    from kibitzer import KibitzerSession
    _HAS_KIBITZER = True
except ImportError:
    _HAS_KIBITZER = False


class LackpyService:
    """Unified service layer orchestrating the lackpy pipeline.

    Both the MCP server and CLI are thin adapters over this class.
    Initialize with a workspace path to set the working directory
    for tool execution.

    When kibitzer is installed, the service automatically creates a
    KibitzerSession for mode enforcement, tool tracking, and coaching.

    Args:
        workspace: Root directory for tool execution. Defaults to cwd.
        config: Override configuration. Loaded from .lackpy/config.toml if not provided.
    """

    def __init__(self, workspace: Path | None = None, config: LackpyConfig | None = None,
                 harness_resolver: Any = None) -> None:
        self._workspace = workspace or Path.cwd()
        self._config = config or load_config(self._workspace)
        # Optional host/harness resolver for virtual tools: name -> callable | None.
        self._harness_resolver = harness_resolver
        self._runner = RestrictedRunner()
        # Single-flight execution: serializes the process-global os.chdir in
        # _execute so concurrent delegations can't race on cwd once an execution
        # runs off the loop thread (the threaded/async path below).
        self._exec_lock = asyncio.Lock()
        # Bridge for execution-loop-local async tools. NB: MCP tools do NOT use
        # this — they marshal to their own client-loop bridge (McpClient.bridge);
        # this one is set per threaded run for any tool bound to the exec loop.
        self._bridge = AsyncBridge()
        self._mcp_client: Any = None  # lazily created if [mcp_servers] are configured
        self.toolbox = Toolbox()
        # Resolution mechanisms for directly-registered specs (tools themselves
        # come from sources, not these). Kept for back-compat with callers that
        # register their own provider="builtin"/"python" specs.
        self.toolbox.register_provider(BuiltinProvider())
        self.toolbox.register_provider(PythonProvider())
        for source, precedence in self._build_tool_sources():
            self.toolbox.add_source(source, precedence)
        self._inference_providers: list = []
        # The configured LLM model (set by _init_inference_providers from the woollama
        # tier) — recorded explicitly instead of reflecting `getattr(p, "_model")`.
        self._llm_model: str | None = None
        self._init_inference_providers()
        self._policy = PolicyLayer()
        self._policy.add_source(KitPolicySource(self.toolbox))
        self._kibitzer: Any = None
        self._init_kibitzer()

    # Source precedences (higher wins the bare name on collision; see Toolbox).
    _PREC_CONFIG = 20    # user [[tools]]
    _PREC_DEFAULT = 10   # shipped builtins
    _PREC_VIRTUAL = 7    # [[virtual_tools]] (declared locally, harness-implemented)
    _PREC_OWN_MCP = 5    # own [mcp_servers]
    _PREC_HOST_MCP = 4   # host configs (earlier file higher; floors at 1)

    def _build_tool_sources(self) -> list[tuple[Any, float]]:
        """Assemble (source, precedence) pairs that populate the toolbox.

        Precedence: user ``[[tools]]`` > shipped defaults > own ``[mcp_servers]`` >
        host configs (auditable local source wins; see docs/design/tool-sources.md §4).
        """
        from .sources import load_default_tool_defs
        from .sources.config import ConfigToolSource

        sources: list[tuple[Any, float]] = [
            (ConfigToolSource(load_default_tool_defs(), name="default"), self._PREC_DEFAULT),
        ]
        if self._config.tools:
            sources.append((ConfigToolSource(self._config.tools, name="config"), self._PREC_CONFIG))
        if self._config.virtual_tools:
            from .sources.virtual import VirtualToolSource
            sources.append((VirtualToolSource(self._config.virtual_tools, self._harness_resolver),
                            self._PREC_VIRTUAL))
        sources.extend(self._build_mcp_sources())
        return sources

    def _build_mcp_sources(self) -> list[tuple[Any, float]]:
        """McpToolSources for own [mcp_servers] and ingested host configs.

        A server that fails to connect reports available()==False and is skipped
        by add_source — one dead server (or one malformed host config) never
        breaks the others.
        """
        own = self._config.mcp_servers
        host_paths = self._config.mcp_host_configs
        if not own and not host_paths:
            return []
        from .sources.mcp import mcp_available
        if not mcp_available():
            return []
        from .sources.mcp.client import McpServerSpec, McpClient
        from .sources.mcp.host_config import load_host_servers
        from .sources.mcp.source import McpToolSource

        if self._mcp_client is None:
            self._mcp_client = McpClient()
        out: list[tuple[Any, float]] = []

        # own servers — highest MCP precedence
        for server_id, cfg in own.items():
            spec = McpServerSpec(
                server_id=server_id,
                transport=cfg.get("transport", "stdio"),
                command=cfg.get("command"),
                args=cfg.get("args", []),
                env=cfg.get("env"),
                cwd=cfg.get("cwd"),
                url=cfg.get("url"),
                headers=cfg.get("headers"),
            )
            overrides = {
                name: (g["w"], g["d"])
                for name, t in (cfg.get("tools", {}) or {}).items()
                if isinstance((g := (t or {}).get("grade")), dict) and "w" in g and "d" in g
            }
            out.append((McpToolSource(spec, self._mcp_client, grade_overrides=overrides),
                        self._PREC_OWN_MCP))

        # host configs — earlier file wins (descending precedence, floored at 1)
        for i, path in enumerate(host_paths):
            prec = max(1.0, self._PREC_HOST_MCP - i)
            try:
                specs = load_host_servers(path)
            except Exception:
                logger.warning("skipping unreadable MCP host config: %s", path)
                continue
            for spec in specs:
                out.append((McpToolSource(spec, self._mcp_client), prec))
        return out

    def close(self) -> None:
        """Release background resources (the MCP client thread). Idempotent.

        Best-effort: the MCP client runs on a daemon thread, so a one-shot
        process (e.g. a CLI command) exits cleanly without calling this. Call it
        for prompt cleanup in a long-lived host that creates many services.
        """
        if self._mcp_client is not None:
            self._mcp_client.shutdown()
            self._mcp_client = None

    def _init_inference_providers(self) -> None:
        templates_dir = self._config.config_dir / "templates"
        self._inference_providers.append(TemplatesProvider(templates_dir=templates_dir))
        self._inference_providers.append(RulesProvider())
        for name in self._config.inference_order:
            provider_cfg = self._config.inference_providers.get(name, {})
            plugin = provider_cfg.get("plugin", "")
            if plugin in ("ollama", "anthropic") and name not in ("templates", "rules"):
                # Retired in the woollama consolidation: lackpy no longer does
                # per-vendor model HTTP. Route these through the `woollama` plugin
                # with a "<provider>/<model>" model string instead.
                logger.warning(
                    "inference plugin %r was retired; use plugin = \"woollama\" with "
                    "model = \"%s/<model>\" (run `lackpyctl init` to regenerate "
                    "config). Skipping inference tier %r.",
                    plugin, plugin, name,
                )
                continue
            if plugin == "woollama" and name not in ("templates", "rules"):
                # Consolidated model management: one provider routes to ANY
                # woollama-known backend (ollama, anthropic, openai, …) via a
                # "<provider>/<model>" string — lackpy stops doing per-vendor HTTP.
                from .infer.providers.woollama import WoollamaProvider
                model = provider_cfg.get("model", "ollama/qwen2.5-coder:1.5b")
                self._llm_model = model  # recorded for PolicyContext (no _model reflection)
                self._inference_providers.append(WoollamaProvider(
                    model=model,
                    temperature=provider_cfg.get("temperature", 0.2),
                    retry_temperature=provider_cfg.get("retry_temperature", 0.6),
                    api_key=provider_cfg.get("api_key"),
                    base_url=provider_cfg.get("base_url"),
                    options=provider_cfg.get("options"),
                    params=provider_cfg.get("params"),
                ))
            elif plugin == "cascade" and name not in ("templates", "rules"):
                from .infer.providers.cascade import CascadeProvider
                self._inference_providers.append(CascadeProvider(
                    host=provider_cfg.get("host", "http://localhost:11434"),
                    tiers=provider_cfg.get("tiers"),
                ))

    def _providers_for(self, model: str | None, temperature: float | None) -> list:
        """The inference providers for one call, applying a per-call model/temperature.

        No override → the shared global list (zero allocation, identical behavior —
        invariant 8). With an override → a copy where each tier that supports it (the
        woollama LLM tier, via ``with_overrides``) is cloned to the requested
        model/temperature; deterministic tiers (templates/rules) pass through unchanged.
        """
        if model is None and temperature is None:
            return self._inference_providers
        return [
            p.with_overrides(model=model, temperature=temperature)
            if hasattr(p, "with_overrides") else p
            for p in self._inference_providers
        ]

    def _init_kibitzer(self) -> None:
        """Initialize Kibitzer session if available."""
        if not _HAS_KIBITZER:
            return
        try:
            self._kibitzer = KibitzerSession(project_dir=self._workspace)
            self._kibitzer.load()
            self._kibitzer.register_tools([
                {
                    "name": spec.name,
                    "grade": {"w": spec.grade_w, "d": spec.effects_ceiling},
                    "description": spec.description,
                    "effects": "write" if spec.grade_w >= 3 else "read" if spec.grade_w >= 1 else "none",
                }
                for spec in self.toolbox.list_tools()
            ])
            self._register_kibitzer_docs()
            from .policy.sources.kibitzer import KibitzerPolicySource
            self._policy.add_source(KibitzerPolicySource(self._kibitzer))
        except Exception:
            self._kibitzer = None

    def _register_kibitzer_docs(self) -> None:
        """Register tool docs with Kibitzer for contextual retrieval."""
        register = getattr(self._kibitzer, "register_docs", None)
        if register is None:
            return
        try:
            from kibitzer import DocRefinement
            from .infer.distill import build_doc_refinement
            docs = self.docs_index()
            register(
                docs["tool_docs"],
                docs_root=str(self._workspace),
                refinement=build_doc_refinement(),
            )
        except Exception:
            pass

    def _resolve_profile(self, profile: "Profile | str | list[str] | dict | None",
                         extra_tools: list[str] | None = None) -> ResolvedProfile:
        """Resolve a profile (name, bundle, or bare tool selection) to a ResolvedProfile.

        Composes the source-populated toolbox + the configured ``[profiles.<name>]``
        tables; ``None`` → the configured default profile. The degenerate (tools-only)
        case is exactly the former kit resolution — ``.tools`` is the same ResolvedKit.
        """
        if profile is None:
            profile = self._config.profile_default
        kits_dir = self._config.config_dir / "kits"
        return resolve_profile(profile, self.toolbox, profiles=self._config.profiles,
                               kits_dir=kits_dir, extra_tools=extra_tools)

    def _gate_kit(self, resolved: ResolvedKit) -> ResolvedKit:
        """Generation gate: drop virtual tools the harness can't currently supply.

        Keeps the model from composing against an absent tool. (The call-time
        proxy still raises if a tool is withdrawn between generation and call.)
        Non-virtual tools are never gated.
        """
        drop = {
            n for n, s in resolved.tools.items()
            if s.provider == "virtual"
            and (self._harness_resolver is None or self._harness_resolver(n) is None)
        }
        if not drop:
            return resolved
        from .lang.grader import compute_grade
        tools = {n: s for n, s in resolved.tools.items() if n not in drop}
        callables = {n: c for n, c in resolved.callables.items() if n not in drop}
        grade = compute_grade(
            {n: {"grade_w": s.grade_w, "effects_ceiling": s.effects_ceiling} for n, s in tools.items()}
        )
        return ResolvedKit(
            tools=tools, callables=callables, grade=grade,
            description=self.toolbox.format_description(list(tools.keys())),
            docs=resolved.docs,
        )

    def _resolve_params(self, params: dict[str, Any] | None, kit: ResolvedKit) -> tuple[dict[str, Any], str | None, set[str]]:
        if not params:
            return {}, None, set()
        values: dict[str, Any] = {}
        for name, val in params.items():
            if isinstance(val, dict) and "value" in val:
                values[name] = val["value"]
            else:
                values[name] = val
        collisions = set(values.keys()) & (set(kit.tools.keys()) | ALLOWED_BUILTINS)
        if collisions:
            raise ValueError(f"Param names collide with tool/builtin names: {collisions}")
        params_desc = format_params_description(params)
        return values, params_desc, set(values.keys())

    def validate(self, program: str, profile: Profile | str | list[str] | dict | None = None,
                 rules: list | None = None, param_names: set[str] | None = None,
                 extra_tools: list[str] | None = None) -> ValidationResult:
        """Validate a lackpy program against a kit's allowed names.

        Args:
            program: The lackpy program source to validate.
            profile: Profile name, bundle, tool list, or dict mapping. Defaults to config default.
            rules: Additional validation rules to apply beyond core checks.
            param_names: Extra names (e.g. parameter names) to allow in the program.

        Returns:
            A ValidationResult indicating whether the program is valid.
        """
        resolved = self._resolve_profile(profile, extra_tools=extra_tools).tools
        allowed = set(resolved.tools.keys())
        if param_names:
            allowed |= param_names
        return validate(program, allowed_names=allowed, extra_rules=rules)

    async def generate(self, intent: str, profile: Profile | str | list[str] | dict | None = None,
                       params: dict[str, Any] | None = None, rules: list | None = None,
                       mode: str | None = None, interpreter: Any = None,
                       extra_tools: list[str] | None = None,
                       model: str | None = None,
                       temperature: float | None = None) -> GenerationResult:
        """Generate a lackpy program from a natural language intent.

        Args:
            intent: Natural language description of the desired program.
            profile: Profile name, bundle, tool list, or dict mapping.
            params: Named input values available to the generated program.
            rules: Additional validation rules the generated program must satisfy.
            mode: Inference strategy mode (e.g. '1-shot', 'spm'). Defaults to config or '1-shot'.
            interpreter: An interpreter instance whose ``system_prompt_hint()``
                is forwarded to the prompt builder. When present, the inference
                prompt uses interpreter-specialized framing instead of the
                generic Jupyter-cell template.
            model: Per-call LLM model override (``"<provider>/<model>"``). When set, the
                woollama tier is cloned to this model for this call only; when omitted,
                the service-configured model is used (identical to prior behavior). This
                is how a profile selects a per-task model (RFC 0002 incr. 5).
            temperature: Per-call temperature override for the LLM tier.

        Returns:
            A GenerationResult with the generated program and provider metadata.

        Raises:
            RuntimeError: If all inference providers fail to produce a valid program.
        """
        from .infer.strategy import STRATEGIES
        from .infer.context import StepContext

        rp = self._resolve_profile(profile, extra_tools=extra_tools)
        resolved = self._gate_kit(rp.tools)
        _, params_desc, param_names = self._resolve_params(params, resolved)

        # Inference selections: an explicit call arg wins over the profile's; both fall
        # back to the service config (woollama tier / inference_mode) downstream.
        effective_model = model if model is not None else rp.model
        effective_temp = temperature if temperature is not None else rp.temperature
        # Per-call providers: the global set, or a copy with the LLM tier overridden when a
        # model/temperature is selected. Neither selected → identical to before (invariant 8).
        providers = self._providers_for(effective_model, effective_temp)

        effective_mode = mode or rp.mode or self._config.inference_mode

        if effective_mode and effective_mode in STRATEGIES:
            strategy_cls = STRATEGIES[effective_mode]
            strategy = strategy_cls()
            dispatcher = InferenceDispatcher(providers=providers)
            # SPM needs an LLM provider (skip deterministic templates/rules)
            if effective_mode == "spm":
                llm_providers = [p for p in dispatcher.get_providers()
                                 if p.name not in ("templates", "rules")]
                if not llm_providers:
                    raise RuntimeError("SPM mode requires an LLM provider (e.g. ollama, anthropic)")
                provider = llm_providers[0]
            else:
                provider = dispatcher.get_provider()
            step = strategy.build(provider)
            ctx = StepContext(
                intent=intent, kit=resolved,
                params_desc=params_desc, extra_rules=rules,
            )
            ctx = await step.run(ctx)
            if ctx.current and ctx.current.valid:
                return GenerationResult(
                    program=ctx.current.program,
                    provider_name=ctx.current.trace.provider_name or "unknown",
                    generation_time_ms=sum(p.trace.duration_ms for p in ctx.programs),
                    correction_strategy=ctx.current.trace.step_name if len(ctx.programs) > 2 else None,
                    correction_attempts=max(0, len(ctx.programs) - 1),
                )
            raise RuntimeError(
                f"Strategy '{effective_mode}' failed. "
                f"Last errors: {ctx.current.errors if ctx.current else 'no programs generated'}"
            )

        # Default: legacy dispatcher path
        dispatcher = InferenceDispatcher(providers=providers)
        allowed = set(resolved.tools.keys()) | param_names

        # Resolve policy — kit baseline + optional kibitzer hints + optional umwelt.
        # The active model is known explicitly (per-call override or the configured LLM
        # model) — no more reflecting `getattr(provider, "_model")`.
        from .policy.types import PolicyContext, ModelSpec
        model_name = effective_model or self._llm_model
        policy_context: PolicyContext = {"kit": resolved}
        if model_name:
            policy_context["model"] = ModelSpec(name=model_name)
        # Thread the active operating mode (if kibitzer is tracking one) so umwelt's
        # mode-scoped policy resolves to the mode the agent is actually in.
        active_mode = getattr(self._kibitzer, "mode", None)
        if active_mode:
            policy_context["mode"] = active_mode
        policy = self._policy.resolve(policy_context)
        namespace_desc = policy.namespace_desc or resolved.description

        return await dispatcher.generate(
            intent=intent, namespace_desc=namespace_desc,
            allowed_names=allowed, params_desc=params_desc, extra_rules=rules,
            interpreter=interpreter, kibitzer_session=self._kibitzer,
        )

    async def _execute(self, program: str, callables: dict[str, Any],
                       param_values: dict[str, Any], kibitzer_session: Any = None) -> ExecutionResult:
        """Run a (already validated) program through the restricted runner.

        The single execution path shared by ``run_program`` and ``delegate``.
        Held under ``_exec_lock`` so the process-global ``os.chdir`` is
        single-flight regardless of which path runs.

        - **Inline** (default): no loop-bound tools in the kit — run the sync
          runner directly, exactly as before.
        - **Threaded**: a tool is flagged loop-bound (e.g. an MCP proxy) — run the
          sync runner via ``run_in_executor`` so the event loop stays free to
          service coroutines those tools marshal back via :class:`AsyncBridge`.
        """
        needs_loop = any(is_async_callable(fn) for fn in callables.values())
        async with self._exec_lock:
            if not needs_loop:
                return self._run_sync(program, callables, param_values, kibitzer_session)
            loop = asyncio.get_running_loop()
            self._bridge.loop = loop
            try:
                return await loop.run_in_executor(
                    None,
                    lambda: self._run_sync(program, callables, param_values, kibitzer_session),
                )
            finally:
                self._bridge.loop = None

    def _run_sync(self, program: str, callables: dict[str, Any],
                  param_values: dict[str, Any], kibitzer_session: Any = None) -> ExecutionResult:
        """Switch into the workspace, run the program, restore cwd. Always sync."""
        prev_cwd = os.getcwd()
        try:
            os.chdir(self._workspace)
            return self._runner.run(
                program, callables, params=param_values,
                kibitzer_session=kibitzer_session,
            )
        finally:
            os.chdir(prev_cwd)

    async def run_program(self, program: str, profile: Profile | str | list[str] | dict | None = None,
                          params: dict[str, Any] | None = None, sandbox: Any = None,
                          rules: list | None = None,
                          extra_tools: list[str] | None = None) -> ExecutionResult:
        """Validate and execute a lackpy program.

        Validates the program before running it; returns a failed ExecutionResult
        immediately if validation fails rather than raising.

        Args:
            program: The lackpy program source to execute.
            profile: Profile name, bundle, tool list, or dict mapping.
            params: Named input values injected into the execution namespace.
            sandbox: Reserved for future sandbox configuration (unused).
            rules: Additional validation rules to apply before execution.

        Returns:
            An ExecutionResult with output, trace, and success status.
        """
        resolved = self._gate_kit(self._resolve_profile(profile, extra_tools=extra_tools).tools)
        param_values, _, param_names = self._resolve_params(params, resolved)
        allowed = set(resolved.tools.keys()) | param_names
        validation = validate(program, allowed_names=allowed, extra_rules=rules)
        if not validation.valid:
            return ExecutionResult(success=False, error=f"Validation failed: {'; '.join(validation.errors)}")
        return await self._execute(program, resolved.callables, param_values)

    async def delegate(self, intent: str, profile: Profile | str | list[str] | dict | None = None,
                       params: dict[str, Any] | None = None, sandbox: Any = None,
                       rules: list | None = None,
                       _program_override: str | None = None,
                       mode: str | None = None,
                       interpreter: Any = None,
                       extra_tools: list[str] | None = None,
                       model: str | None = None,
                       temperature: float | None = None) -> dict[str, Any]:
        """Generate and execute a program from a natural language intent in one step.

        Combines generate and run_program: generates a program from intent, then
        executes it, returning a combined result dict with timing and trace details.

        Args:
            intent: Natural language description of the desired program.
            profile: Profile name, bundle, tool list, or dict mapping.
            params: Named input values available to the program.
            sandbox: Reserved for future sandbox configuration (unused).
            rules: Additional validation rules for generation and execution.
            interpreter: An interpreter instance whose ``system_prompt_hint()``
                is forwarded to the prompt builder for interpreter-specialized
                framing. When omitted, the default Jupyter-cell prompt is used.

        Returns:
            A dict with keys: success, program, grade, generation_tier,
            generation_time_ms, execution_time_ms, total_time_ms, trace,
            files_read, files_modified, output, error.

        Raises:
            RuntimeError: If all inference providers fail to produce a valid program.
        """
        start = time.perf_counter()
        rp = self._resolve_profile(profile, extra_tools=extra_tools)
        resolved = self._gate_kit(rp.tools)
        # The model actually used (per-call override or the profile's, else configured) —
        # the same value `generate` uses, so kibitzer logs the real model, not a reflection.
        effective_model = (model if model is not None else rp.model) or self._llm_model
        param_values, params_desc, param_names = self._resolve_params(params, resolved)

        # Kibitzer: register context for this delegation
        if self._kibitzer:
            self._kibitzer.register_context({
                "source": "lackpy",
                "intent": intent,
                "kit": list(resolved.tools.keys()),
            })

        if _program_override:
            from .infer.dispatch import GenerationResult
            gen_result = GenerationResult(
                program=_program_override,
                provider_name="lackey_file",
                generation_time_ms=0.0,
            )
        else:
            gen_result = await self.generate(intent, profile, params, rules, mode=mode,
                                            interpreter=interpreter, extra_tools=extra_tools,
                                            model=model, temperature=temperature)

        # Kibitzer: validate planned calls before execution
        kibitzer_suggestions: list[str] = []
        if self._kibitzer:
            validation_result = validate(gen_result.program, allowed_names=set(resolved.tools.keys()) | param_names)
            if validation_result.valid:
                planned = [{"tool": call, "input": {}} for call in validation_result.calls
                           if call not in ALLOWED_BUILTINS]
                violations = self._kibitzer.validate_calls(planned)
                if violations:
                    return {
                        "success": False,
                        "program": gen_result.program,
                        "grade": {"w": resolved.grade.w, "d": resolved.grade.d},
                        "generation_tier": gen_result.provider_name,
                        "error": "Kibitzer mode policy violation: " + "; ".join(v.reason for v in violations),
                        "correction_strategy": gen_result.correction_strategy,
                        "correction_attempts": gen_result.correction_attempts,
                    }

        exec_result = await self._execute(
            gen_result.program, resolved.callables, param_values,
            kibitzer_session=self._kibitzer,
        )

        # Kibitzer: get coaching suggestions after execution
        if self._kibitzer:
            kibitzer_suggestions = self._kibitzer.get_suggestions()
            # Classify failure mode for kibitzer's pattern tracker
            failure_mode = None
            if not exec_result.success:
                from .infer.failure_modes import classify_failure
                validation_result = validate(
                    gen_result.program,
                    allowed_names=set(resolved.tools.keys()) | param_names,
                )
                failure_mode = classify_failure(
                    gate_passed=validation_result.valid,
                    gate_errors=list(validation_result.errors),
                    exec_error=exec_result.error,
                    sanitized_program=gen_result.program,
                )
            # Report generation outcome with extended fields
            interp_name = getattr(interpreter, "name", None) if interpreter else None
            self._kibitzer.report_generation({
                "intent": intent,
                "program": gen_result.program,
                "provider": gen_result.provider_name,
                "model": effective_model,
                "correction_attempts": gen_result.correction_attempts,
                "correction_strategy": gen_result.correction_strategy,
                "success": exec_result.success,
                "interpreter": interp_name,
                "prompt_specialized": interp_name is not None,
                "failure_mode": failure_mode,
            })
            self._kibitzer.save()

        total_ms = (time.perf_counter() - start) * 1000
        result = {
            "success": exec_result.success,
            "program": gen_result.program,
            "grade": {"w": resolved.grade.w, "d": resolved.grade.d},
            "generation_tier": gen_result.provider_name,
            "generation_time_ms": gen_result.generation_time_ms,
            "execution_time_ms": total_ms - gen_result.generation_time_ms,
            "total_time_ms": total_ms,
            "trace": [{"step": e.step, "tool": e.tool, "args": e.args, "result": e.result,
                        "duration_ms": e.duration_ms, "success": e.success, "error": e.error}
                       for e in exec_result.trace.entries],
            "files_read": exec_result.trace.files_read,
            "files_modified": exec_result.trace.files_modified,
            # Prefer the typed last-expression value; if the program instead
            # printed its answer (a common small-model habit), fall back to the
            # captured stdout so the result isn't silently dropped. stdout is
            # always surfaced so the caller can see it regardless.
            "output": exec_result.effective_output,
            "stdout": exec_result.stdout,
            "error": exec_result.error,
            "correction_strategy": gen_result.correction_strategy,
            "correction_attempts": gen_result.correction_attempts,
        }
        if kibitzer_suggestions:
            result["kibitzer_suggestions"] = kibitzer_suggestions

        # Best-effort: record this delegation as a blq invocation when the
        # workspace uses blq (.bird/ present). No-op otherwise; never raises.
        from .observ import record_delegation
        record_delegation(self._workspace, intent, result)

        return result

    def parse_lackey(self, path: Path) -> Any:
        """Parse a Lackey file and extract metadata."""
        from .lackey.parser import parse_lackey as _parse
        return _parse(path)

    async def run_lackey(
        self, path: Path,
        params: dict[str, Any] | None = None,
        sandbox: Any = None,
    ) -> dict[str, Any]:
        """Load and run a Lackey file."""
        from .lackey.parser import parse_lackey as _parse
        import ast as _ast

        info = _parse(path)

        merged_params: dict[str, Any] = {}
        for name, spec in info.params.items():
            if params and name in params:
                merged_params[name] = params[name]
            elif "default" in spec:
                merged_params[name] = spec["default"]

        tools = info.tools

        # Convert top-level `return X` to bare expression `X` so the runner can
        # execute the body as a flat program (return is only valid inside a function).
        run_body = _strip_top_level_return(info.run_body)

        return await self.delegate(
            intent="",
            profile=tools,
            params=merged_params if merged_params else None,
            _program_override=run_body,
        )

    async def create_lackey(
        self,
        program: str,
        name: str,
        tools: list[str],
        params: dict[str, dict] | None = None,
        returns: str | None = None,
        creation_log: list[dict] | None = None,
        output_dir: Path | None = None,
    ) -> Path:
        """Wrap a generated program in a Lackey class and save."""
        from .lackey.creator import save_lackey

        if output_dir is None:
            output_dir = self._config.config_dir / "templates"
            output_dir.mkdir(parents=True, exist_ok=True)

        return save_lackey(
            program=program, name=name, tools=tools,
            output_dir=output_dir, params=params,
            returns=returns, creation_log=creation_log,
        )

    async def create(self, program: str, profile: Profile | str | list[str] | dict | None = None,
                     name: str = "", pattern: str | None = None,
                     extra_tools: list[str] | None = None) -> dict[str, Any]:
        """Validate a program and save it as a named template.

        Args:
            program: The lackpy program source to save.
            kit: Kit name, list of tool names, or dict mapping used for validation.
            name: Template name (used as the filename stem).
            pattern: Optional intent pattern string for template matching.

        Returns:
            A dict with keys: success (bool), path (str) on success,
            or errors (list) on validation failure.
        """
        resolved = self._resolve_profile(profile, extra_tools=extra_tools).tools
        validation = validate(program, allowed_names=set(resolved.tools.keys()))
        if not validation.valid:
            return {"success": False, "errors": validation.errors}
        templates_dir = self._config.config_dir / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        template_file = templates_dir / f"{name}.tmpl"
        content = f"---\nname: {name}\n"
        if pattern:
            content += f'pattern: "{pattern}"\n'
        content += "success_count: 0\nfail_count: 0\n---\n" + program
        template_file.write_text(content)
        return {"success": True, "path": str(template_file)}

    def profile_info(self, profile: Profile | str | list[str] | dict,
                     extra_tools: list[str] | None = None) -> dict[str, Any]:
        """Return metadata for a resolved profile (its tools, grade, description).

        Args:
            profile: Profile name, bundle, tool list, or dict mapping.

        Returns:
            A dict with keys: tools (mapping of tool name to spec dict),
            grade (w and d values), and description (formatted namespace string).

        Raises:
            KeyError: If the kit references an unknown tool.
            FileNotFoundError: If a named kit file does not exist.
        """
        resolved = self._resolve_profile(profile, extra_tools=extra_tools).tools
        return {
            "tools": {name: {"description": spec.description, "grade_w": spec.grade_w,
                             "effects_ceiling": spec.effects_ceiling}
                      for name, spec in resolved.tools.items()},
            "grade": {"w": resolved.grade.w, "d": resolved.grade.d},
            "description": resolved.description,
        }

    def profile_list(self) -> list[dict[str, str]]:
        """List named tool-set files in the workspace configuration directory.

        Returns:
            A list of dicts with keys: name (stem) and path (absolute path string).
            Returns an empty list if the directory does not exist.
        """
        kits_dir = self._config.config_dir / "kits"
        if not kits_dir.exists():
            return []
        files = sorted([*kits_dir.glob("*.profile"), *kits_dir.glob("*.kit")])
        return [{"name": p.stem, "path": str(p)} for p in files]

    def profile_create(self, name: str, tools: list[str], description: str | None = None) -> dict[str, Any]:
        """Create a named tool-set file in the workspace configuration directory.

        Args:
            name: Name used as the filename stem.
            tools: List of tool names to include.
            description: Optional human-readable description written to the frontmatter.

        Returns:
            A dict with keys: name, path (absolute path string), and tools.
        """
        kits_dir = self._config.config_dir / "kits"
        kits_dir.mkdir(parents=True, exist_ok=True)
        kit_file = kits_dir / f"{name}.kit"
        content = f"---\nname: {name}\n"
        if description:
            content += f"description: {description}\n"
        content += "---\n" + "\n".join(tools) + "\n"
        kit_file.write_text(content)
        return {"name": name, "path": str(kit_file), "tools": tools}

    def docs_index(self, profile: Profile | str | list[str] | dict | None = None,
                   extra_tools: list[str] | None = None) -> dict[str, Any]:
        """Return documentation references for a kit's tools.

        Returns a dict with ``tool_docs`` (tool name → relative doc path)
        and ``kit_docs`` (list of kit-level doc paths). Paths are relative
        to the package/workspace root — callers resolve them on demand.
        """
        resolved = self._resolve_profile(profile, extra_tools=extra_tools).tools
        tool_docs = {name: spec.docs for name, spec in resolved.tools.items() if spec.docs}
        return {
            "tool_docs": tool_docs,
            "kit_docs": resolved.docs,
        }

    def resolve_doc(self, doc_path: str) -> str | None:
        """Read a documentation file by its relative path.

        Resolves the path against the workspace root and returns the
        content, or None if the file does not exist.
        """
        resolved = self._workspace / doc_path
        if resolved.exists():
            return resolved.read_text()
        return None

    def toolbox_list(self) -> list[dict[str, Any]]:
        """List all registered tools across all providers.

        Returns:
            A list of dicts with keys: name, provider, description,
            grade_w, and effects_ceiling for each registered tool.
        """
        return [{"name": s.name, "provider": s.provider, "description": s.description,
                 "grade_w": s.grade_w, "effects_ceiling": s.effects_ceiling}
                for s in self.toolbox.list_tools()]

    def get_config(self) -> dict[str, Any]:
        """Return a serializable snapshot of the workspace configuration."""
        return {
            "workspace": str(self._workspace),
            "config_dir": str(self._config.config_dir),
            "inference_order": self._config.inference_order,
            "inference_mode": self._config.inference_mode,
            "profile_default": self._config.profile_default,
            "sandbox_enabled": self._config.sandbox_enabled,
            "sandbox_timeout": self._config.sandbox_timeout,
            "sandbox_memory_mb": self._config.sandbox_memory_mb,
            "tools": len(self.toolbox.tools),
        }

    def provider_list(self) -> list[dict[str, Any]]:
        """List configured inference providers with availability status."""
        result = []
        for provider in self._inference_providers:
            entry: dict[str, Any] = {
                "name": provider.name,
                "available": provider.available(),
            }
            # Add config details for providers that have them
            provider_cfg = self._config.inference_providers.get(provider.name, {})
            entry["plugin"] = provider_cfg.get("plugin", provider.name)
            if "model" in provider_cfg:
                entry["model"] = provider_cfg["model"]
            if "host" in provider_cfg:
                entry["host"] = provider_cfg["host"]
            if "temperature" in provider_cfg:
                entry["temperature"] = provider_cfg["temperature"]
            result.append(entry)
        return result

    def language_spec(self) -> dict[str, Any]:
        """Return the lackpy language specification as structured data."""
        from .lang.spec import get_spec
        return get_spec()
