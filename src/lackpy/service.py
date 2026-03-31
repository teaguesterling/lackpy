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
from .kit.toolbox import Toolbox, ToolSpec
from .lang.grammar import ALLOWED_BUILTINS
from .lang.validator import ValidationResult, validate
from .run.base import ExecutionResult
from .run.runner import RestrictedRunner


class LackpyService:
    def __init__(self, workspace: Path | None = None, config: LackpyConfig | None = None) -> None:
        self._workspace = workspace or Path.cwd()
        self._config = config or load_config(self._workspace)
        self._runner = RestrictedRunner()
        self.toolbox = Toolbox()
        self.toolbox.register_provider(BuiltinProvider())
        self.toolbox.register_provider(PythonProvider())
        self._inference_providers: list = []
        self._init_inference_providers()

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
        resolved = self._resolve_kit(kit)
        allowed = set(resolved.tools.keys())
        if param_names:
            allowed |= param_names
        return validate(program, allowed_names=allowed, extra_rules=rules)

    async def generate(self, intent: str, kit: str | list[str] | dict | None = None,
                       params: dict[str, Any] | None = None, rules: list | None = None) -> GenerationResult:
        resolved = self._resolve_kit(kit)
        _, params_desc, param_names = self._resolve_params(params, resolved)
        dispatcher = InferenceDispatcher(providers=self._inference_providers)
        allowed = set(resolved.tools.keys()) | param_names
        return await dispatcher.generate(
            intent=intent, namespace_desc=resolved.description,
            allowed_names=allowed, params_desc=params_desc, extra_rules=rules,
        )

    async def run_program(self, program: str, kit: str | list[str] | dict | None = None,
                          params: dict[str, Any] | None = None, sandbox: Any = None,
                          rules: list | None = None) -> ExecutionResult:
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
                       rules: list | None = None) -> dict[str, Any]:
        start = time.perf_counter()
        resolved = self._resolve_kit(kit)
        param_values, params_desc, param_names = self._resolve_params(params, resolved)
        gen_result = await self.generate(intent, kit, params, rules)
        prev_cwd = os.getcwd()
        try:
            os.chdir(self._workspace)
            exec_result = self._runner.run(gen_result.program, resolved.callables, params=param_values)
        finally:
            os.chdir(prev_cwd)
        total_ms = (time.perf_counter() - start) * 1000
        return {
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
        }

    async def create(self, program: str, kit: str | list[str] | dict | None = None,
                     name: str = "", pattern: str | None = None) -> dict[str, Any]:
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
        resolved = self._resolve_kit(kit)
        return {
            "tools": {name: {"description": spec.description, "grade_w": spec.grade_w,
                             "effects_ceiling": spec.effects_ceiling}
                      for name, spec in resolved.tools.items()},
            "grade": {"w": resolved.grade.w, "d": resolved.grade.d},
            "description": resolved.description,
        }

    def kit_list(self) -> list[dict[str, str]]:
        kits_dir = self._config.config_dir / "kits"
        if not kits_dir.exists():
            return []
        return [{"name": p.stem, "path": str(p)} for p in sorted(kits_dir.glob("*.kit"))]

    def kit_create(self, name: str, tools: list[str], description: str | None = None) -> dict[str, Any]:
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
        return [{"name": s.name, "provider": s.provider, "description": s.description,
                 "grade_w": s.grade_w, "effects_ceiling": s.effects_ceiling}
                for s in self.toolbox.list_tools()]
