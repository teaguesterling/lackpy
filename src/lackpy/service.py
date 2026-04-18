"""Unified service layer: the API that MCP and CLI both call."""

from __future__ import annotations

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
from .kit.registry import ResolvedKit, resolve_kit
from .kit.toolbox import ArgSpec, Toolbox, ToolSpec
from .lang.grammar import ALLOWED_BUILTINS

_BUILTIN_TOOLS = [
    ToolSpec(
        name="read_file", provider="builtin", description="Read file contents",
        args=[ArgSpec(name="path", type="str", description="File path")],
        returns="str", grade_w=1, effects_ceiling=1,
    ),
    ToolSpec(
        name="find_files", provider="builtin", description="Find files matching a glob pattern",
        args=[ArgSpec(name="pattern", type="str", description="Glob pattern")],
        returns="list[str]", grade_w=1, effects_ceiling=1,
    ),
    ToolSpec(
        name="write_file", provider="builtin", description="Write content to a file",
        args=[ArgSpec(name="path", type="str"), ArgSpec(name="content", type="str")],
        returns="bool", grade_w=3, effects_ceiling=3,
    ),
    ToolSpec(
        name="edit_file", provider="builtin", description="Replace text in a file",
        args=[ArgSpec(name="path", type="str"), ArgSpec(name="old_str", type="str"), ArgSpec(name="new_str", type="str")],
        returns="bool", grade_w=3, effects_ceiling=3,
    ),
]
from .lang.validator import ValidationResult, validate
from .run.base import ExecutionResult
from .run.runner import RestrictedRunner


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

    def __init__(self, workspace: Path | None = None, config: LackpyConfig | None = None) -> None:
        self._workspace = workspace or Path.cwd()
        self._config = config or load_config(self._workspace)
        self._runner = RestrictedRunner()
        self.toolbox = Toolbox()
        self.toolbox.register_provider(BuiltinProvider())
        self.toolbox.register_provider(PythonProvider())
        for spec in _BUILTIN_TOOLS:
            self.toolbox.register_tool(spec)
        self._inference_providers: list = []
        self._init_inference_providers()
        self._kibitzer: Any = None
        self._init_kibitzer()

    def _init_inference_providers(self) -> None:
        templates_dir = self._config.config_dir / "templates"
        self._inference_providers.append(TemplatesProvider(templates_dir=templates_dir))
        self._inference_providers.append(RulesProvider())
        for name in self._config.inference_order:
            provider_cfg = self._config.inference_providers.get(name, {})
            plugin = provider_cfg.get("plugin", "")
            if plugin == "ollama" and name not in ("templates", "rules"):
                from .infer.providers.ollama import OllamaProvider
                self._inference_providers.append(OllamaProvider(
                    host=provider_cfg.get("host", "http://localhost:11434"),
                    model=provider_cfg.get("model", "qwen2.5-coder:1.5b"),
                    temperature=provider_cfg.get("temperature", 0.2),
                    keep_alive=provider_cfg.get("keep_alive", "30m"),
                ))
            elif plugin == "anthropic" and name not in ("templates", "rules"):
                from .infer.providers.anthropic import AnthropicProvider
                self._inference_providers.append(AnthropicProvider(
                    model=provider_cfg.get("model", "claude-haiku-4-5-20251001"),
                ))
            elif plugin == "cascade" and name not in ("templates", "rules"):
                from .infer.providers.cascade import CascadeProvider
                self._inference_providers.append(CascadeProvider(
                    host=provider_cfg.get("host", "http://localhost:11434"),
                    tiers=provider_cfg.get("tiers"),
                ))

    def _init_kibitzer(self) -> None:
        """Initialize Kibitzer session if available."""
        if not _HAS_KIBITZER:
            return
        try:
            self._kibitzer = KibitzerSession(project_dir=self._workspace)
            self._kibitzer.load()
            # Register our tools so Kibitzer can make grade-aware decisions
            self._kibitzer.register_tools([
                {
                    "name": spec.name,
                    "grade": {"w": spec.grade_w, "d": spec.effects_ceiling},
                    "description": spec.description,
                    "effects": "write" if spec.grade_w >= 3 else "read" if spec.grade_w >= 1 else "none",
                }
                for spec in self.toolbox.list_tools()
            ])
        except Exception:
            self._kibitzer = None

    def _apply_kibitzer_hints(self, namespace_desc: str, model: str | None = None) -> str:
        """Query kibitzer for prompt hints and append them to namespace_desc.

        Called before generation when kibitzer is available. When a model
        name is provided, hints are filtered to patterns observed for that
        specific model — giving model-specific constraints.

        Falls back gracefully if get_prompt_hints() is not implemented
        (older kibitzer version) or returns empty.
        """
        get_hints = getattr(self._kibitzer, "get_prompt_hints", None)
        if get_hints is None:
            return namespace_desc
        try:
            hints = get_hints(model=model)
        except Exception:
            return namespace_desc
        if not hints:
            return namespace_desc
        hint_lines = ["\nDynamic constraints (from observed failure patterns):"]
        for hint in hints:
            if isinstance(hint, dict):
                content = hint.get("content", "")
                if content:
                    hint_lines.append(f"  - {content}")
            elif isinstance(hint, str):
                hint_lines.append(f"  - {hint}")
        if len(hint_lines) > 1:
            return namespace_desc + "\n".join(hint_lines)
        return namespace_desc

    def _resolve_kit(self, kit: str | list[str] | dict | None) -> ResolvedKit:
        if kit is None:
            kit = self._config.kit_default
        kits_dir = self._config.config_dir / "kits"
        return resolve_kit(kit, self.toolbox, kits_dir=kits_dir)

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

    def validate(self, program: str, kit: str | list[str] | dict | None = None,
                 rules: list | None = None, param_names: set[str] | None = None) -> ValidationResult:
        """Validate a lackpy program against a kit's allowed names.

        Args:
            program: The lackpy program source to validate.
            kit: Kit name, list of tool names, or dict mapping. Defaults to config default.
            rules: Additional validation rules to apply beyond core checks.
            param_names: Extra names (e.g. parameter names) to allow in the program.

        Returns:
            A ValidationResult indicating whether the program is valid.
        """
        resolved = self._resolve_kit(kit)
        allowed = set(resolved.tools.keys())
        if param_names:
            allowed |= param_names
        return validate(program, allowed_names=allowed, extra_rules=rules)

    async def generate(self, intent: str, kit: str | list[str] | dict | None = None,
                       params: dict[str, Any] | None = None, rules: list | None = None,
                       mode: str | None = None, interpreter: Any = None) -> GenerationResult:
        """Generate a lackpy program from a natural language intent.

        Args:
            intent: Natural language description of the desired program.
            kit: Kit name, list of tool names, or dict mapping.
            params: Named input values available to the generated program.
            rules: Additional validation rules the generated program must satisfy.
            mode: Inference strategy mode (e.g. '1-shot', 'spm'). Defaults to config or '1-shot'.
            interpreter: An interpreter instance whose ``system_prompt_hint()``
                is forwarded to the prompt builder. When present, the inference
                prompt uses interpreter-specialized framing instead of the
                generic Jupyter-cell template.

        Returns:
            A GenerationResult with the generated program and provider metadata.

        Raises:
            RuntimeError: If all inference providers fail to produce a valid program.
        """
        from .infer.strategy import STRATEGIES
        from .infer.context import StepContext

        resolved = self._resolve_kit(kit)
        _, params_desc, param_names = self._resolve_params(params, resolved)

        effective_mode = mode or self._config.inference_mode

        if effective_mode and effective_mode in STRATEGIES:
            strategy_cls = STRATEGIES[effective_mode]
            strategy = strategy_cls()
            dispatcher = InferenceDispatcher(providers=self._inference_providers)
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
        dispatcher = InferenceDispatcher(providers=self._inference_providers)
        allowed = set(resolved.tools.keys()) | param_names

        # Kibitzer: get model-specific prompt hints from failure pattern tracker
        namespace_desc = resolved.description
        if self._kibitzer:
            # Find the model name from the first available LLM provider
            model_name = None
            for p in dispatcher.get_providers():
                m = getattr(p, "_model", None)
                if m:
                    model_name = m
                    break
            namespace_desc = self._apply_kibitzer_hints(namespace_desc, model=model_name)

        return await dispatcher.generate(
            intent=intent, namespace_desc=namespace_desc,
            allowed_names=allowed, params_desc=params_desc, extra_rules=rules,
            interpreter=interpreter, kibitzer_session=self._kibitzer,
        )

    async def run_program(self, program: str, kit: str | list[str] | dict | None = None,
                          params: dict[str, Any] | None = None, sandbox: Any = None,
                          rules: list | None = None) -> ExecutionResult:
        """Validate and execute a lackpy program.

        Validates the program before running it; returns a failed ExecutionResult
        immediately if validation fails rather than raising.

        Args:
            program: The lackpy program source to execute.
            kit: Kit name, list of tool names, or dict mapping.
            params: Named input values injected into the execution namespace.
            sandbox: Reserved for future sandbox configuration (unused).
            rules: Additional validation rules to apply before execution.

        Returns:
            An ExecutionResult with output, trace, and success status.
        """
        resolved = self._resolve_kit(kit)
        param_values, _, param_names = self._resolve_params(params, resolved)
        allowed = set(resolved.tools.keys()) | param_names
        validation = validate(program, allowed_names=allowed, extra_rules=rules)
        if not validation.valid:
            return ExecutionResult(success=False, error=f"Validation failed: {'; '.join(validation.errors)}")
        prev_cwd = os.getcwd()
        try:
            os.chdir(self._workspace)
            return self._runner.run(program, resolved.callables, params=param_values)
        finally:
            os.chdir(prev_cwd)

    async def delegate(self, intent: str, kit: str | list[str] | dict | None = None,
                       params: dict[str, Any] | None = None, sandbox: Any = None,
                       rules: list | None = None,
                       _program_override: str | None = None,
                       mode: str | None = None,
                       interpreter: Any = None) -> dict[str, Any]:
        """Generate and execute a program from a natural language intent in one step.

        Combines generate and run_program: generates a program from intent, then
        executes it, returning a combined result dict with timing and trace details.

        Args:
            intent: Natural language description of the desired program.
            kit: Kit name, list of tool names, or dict mapping.
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
        resolved = self._resolve_kit(kit)
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
            gen_result = await self.generate(intent, kit, params, rules, mode=mode,
                                            interpreter=interpreter)

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

        prev_cwd = os.getcwd()
        try:
            os.chdir(self._workspace)
            exec_result = self._runner.run(
                gen_result.program, resolved.callables, params=param_values,
                kibitzer_session=self._kibitzer,
            )
        finally:
            os.chdir(prev_cwd)

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
            # Extract model name from the provider that produced the result
            model_name = None
            for p in self._inference_providers:
                if p.name == gen_result.provider_name:
                    model_name = getattr(p, "_model", None)
                    break
            self._kibitzer.report_generation({
                "intent": intent,
                "program": gen_result.program,
                "provider": gen_result.provider_name,
                "model": model_name,
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
            "output": exec_result.output,
            "error": exec_result.error,
            "correction_strategy": gen_result.correction_strategy,
            "correction_attempts": gen_result.correction_attempts,
        }
        if kibitzer_suggestions:
            result["kibitzer_suggestions"] = kibitzer_suggestions
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

        kit = info.tools

        # Convert top-level `return X` to bare expression `X` so the runner can
        # execute the body as a flat program (return is only valid inside a function).
        run_body = _strip_top_level_return(info.run_body)

        return await self.delegate(
            intent="",
            kit=kit,
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

    async def create(self, program: str, kit: str | list[str] | dict | None = None,
                     name: str = "", pattern: str | None = None) -> dict[str, Any]:
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
        resolved = self._resolve_kit(kit)
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

    def kit_info(self, kit: str | list[str] | dict) -> dict[str, Any]:
        """Return metadata for a resolved kit.

        Args:
            kit: Kit name, list of tool names, or dict mapping.

        Returns:
            A dict with keys: tools (mapping of tool name to spec dict),
            grade (w and d values), and description (formatted namespace string).

        Raises:
            KeyError: If the kit references an unknown tool.
            FileNotFoundError: If a named kit file does not exist.
        """
        resolved = self._resolve_kit(kit)
        return {
            "tools": {name: {"description": spec.description, "grade_w": spec.grade_w,
                             "effects_ceiling": spec.effects_ceiling}
                      for name, spec in resolved.tools.items()},
            "grade": {"w": resolved.grade.w, "d": resolved.grade.d},
            "description": resolved.description,
        }

    def kit_list(self) -> list[dict[str, str]]:
        """List all kit files in the workspace configuration directory.

        Returns:
            A list of dicts with keys: name (stem) and path (absolute path string).
            Returns an empty list if the kits directory does not exist.
        """
        kits_dir = self._config.config_dir / "kits"
        if not kits_dir.exists():
            return []
        return [{"name": p.stem, "path": str(p)} for p in sorted(kits_dir.glob("*.kit"))]

    def kit_create(self, name: str, tools: list[str], description: str | None = None) -> dict[str, Any]:
        """Create a new kit file in the workspace configuration directory.

        Args:
            name: Kit name used as the filename stem.
            tools: List of tool names to include in the kit.
            description: Optional human-readable description written to the kit frontmatter.

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
            "kit_default": self._config.kit_default,
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
