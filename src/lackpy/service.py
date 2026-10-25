"""Unified service layer: the API that MCP and CLI both call."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

from .config import LackpyConfig, load_config
from .infer.dispatch import InferenceDispatcher, GenerationResult
from .infer.prompt import format_params_description
from .infer.providers.rules import RulesProvider
from .infer.providers.templates import TemplatesProvider
from .tools.providers.python import PythonProvider
from .tools.registry import ResolvedTools
from .lang.grader import compute_grade
from .profiles import KIT_REMOVED, STAGED_DSL_PENDING, Profile, ProfileError, ResolvedProfile, resolve_profile
from .tools.toolbox import Toolbox
from .lang.grammar import ALLOWED_BUILTINS
from .policy.layer import PolicyLayer
from .policy.sources.tools import ToolsPolicySource
from .lang.validator import ValidationResult, validate
from .run.base import ExecutionResult
from .run.paths import PathPolicy, load_policy, read_only_policy
from .run.process import ProcessRunner, shared_runner
from .run.sessions import SessionError, SessionRegistry
from .run.runner import _binds_params
from .run.refusal import LackpyRefusal
from .run.trace import error_text

logger = logging.getLogger(__name__)


def validation_error(validation: ValidationResult) -> str:
    """The failed-result text for a rejected program (review I3, spec D17).

    Each violation reads ``message (hint)`` when the rule names an approved
    alternative, so the caller learns the path forward from the error alone.
    """
    if validation.violations:
        parts = [f"{v.message} ({v.hint})" if v.hint else v.message for v in validation.violations]
    else:
        parts = list(validation.errors)
    return f"Validation failed: {'; '.join(parts)}"


_RULE_NAME = re.compile(r"^\s*([a-z_]+)\s*(?:\(\s*(\d+)\s*\))?\s*$")


def resolve_rules(rules: list | None) -> list | None:
    """Turn the rule names a caller passes (MCP ``rules: list[str]``) into rule callables.

    A callable is kept as it is. A name is one of ``lackpy.lang.rules``:
    ``no_loops``, ``no_nested_calls``, or ``max_depth(N)`` / ``max_calls(N)``
    with an int ``N`` (review M1). Raises ``ValueError`` naming an unknown or
    malformed rule; each public method turns that into a failed result.
    """
    if not rules:
        return rules
    from .lang import rules as lang_rules
    plain = {"no_loops": lang_rules.no_loops, "no_nested_calls": lang_rules.no_nested_calls}
    factories = {"max_depth": lang_rules.max_depth, "max_calls": lang_rules.max_calls}
    resolved = []
    for rule in rules:
        if callable(rule):
            resolved.append(rule)
            continue
        match = _RULE_NAME.match(rule) if isinstance(rule, str) else None
        name, arg = (match.group(1), match.group(2)) if match else (None, None)
        if name in plain and arg is None:
            resolved.append(plain[name])
        elif name in factories and arg is not None:
            resolved.append(factories[name](int(arg)))
        else:
            known = ", ".join([*plain, *(f"{f}(N)" for f in factories)])
            raise ValueError(f"unknown rule {rule!r}; the known rules are {known}")
    return resolved

def _policy_error(e: BaseException) -> str:
    """The failed-result text for a PathPolicy that cannot be loaded.

    ``load_policy`` raises ``ValueError`` naming the settings file; any other
    type is named too, so a loader defect is not mistaken for a settings one.
    A ``LackpyRefusal`` (a ``BaseException``) is caught by the callers too
    (review I4), so a settings state never escapes to FastMCP.
    """
    if type(e) is ValueError:
        return f"cannot load the path policy: {error_text(e)}"
    return f"cannot load the path policy ({type(e).__name__}): {error_text(e)}"


def _traced_paths(trace: Any, tools: tuple[str, ...]) -> list[str]:
    """The paths, as the program built them, of the successful ``tools`` calls; no repeats."""
    seen: dict[str, None] = {}
    for entry in trace.entries:
        if entry.success and entry.tool in tools and entry.args:
            seen.setdefault(str(entry.args[0]), None)
    return list(seen)


#: The literate grade ceiling of a read-only profile: pinhole reads, no writes.
_READ_ONLY_CEILING = (1, 1)

#: The keys a profile's ``policy`` table may hold.
_POLICY_KEYS = frozenset({"read_only"})


def _read_only(policy: Any) -> bool:
    """A profile's ``policy.read_only`` (default False).

    A read-only profile runs under ``read_only_policy`` (literate stepy D7,
    Ruling F). Fail closed (Task 4 review I2): a ``policy`` that is not a
    table, a key other than ``read_only`` (a typo such as ``readonly``), or a
    value that is not a bool is a ``ProfileError``, never "false".
    """
    if policy is None:
        return False
    if not isinstance(policy, dict):
        raise ProfileError(f"profile policy must be a table such as {{ read_only = true }}, "
                           f"not {policy!r}")
    unknown = sorted(str(k) for k in policy if k not in _POLICY_KEYS)
    if unknown:
        raise ProfileError(f"unknown profile policy key(s) {unknown}; "
                           f"allowed: {sorted(_POLICY_KEYS)}")
    value = policy.get("read_only", False)
    if not isinstance(value, bool):
        raise ProfileError(f"profile policy.read_only must be true or false, not {value!r}")
    return value


def _positive_seconds(value: Any) -> bool:
    """``value`` is an int or float (not a bool), finite and above 0."""
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _check_max_timeout(value: Any) -> None:
    """Fail closed on a profile ``max_timeout`` that is not a number of seconds
    (as ``_read_only`` does for ``policy.read_only``): a TOML string or bool
    would otherwise reach the cap comparison."""
    if value is not None and not _positive_seconds(value):
        raise ProfileError(f"profile max_timeout must be a positive number of seconds, "
                           f"not {value!r}")


def _rule_names(rules: list | None) -> list[str] | None:
    """The rule names the worker resolves again (Plan 3 Task 5).

    A caller's rule callable cannot cross to the worker: the parent validates
    with it, on the same source, before the run, and the worker gets the names.
    """
    names = [r for r in rules or [] if isinstance(r, str)]
    return names or None


def _check_mode(mode: str | None) -> None:
    """Refuse inference mode "staged-dsl" (review N2).

    Its strategy runs ``AstSelectInterpreter`` over the process cwd at generation
    time, with no PathPolicy, and returns the matched source as the program.
    """
    if mode == "staged-dsl":
        raise ProfileError(STAGED_DSL_PENDING)


def _no_tools() -> ResolvedTools:
    """The empty tool set every profile resolves to since the kit was removed."""
    return ResolvedTools(tools={}, callables={}, grade=compute_grade({}), description="")


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


_LITERATE_PREAMBLE_MARKERS = ("here's", "here is", "the following", "the document")


def _strip_literate_wrapper(raw: str) -> str:
    """Literate-safe cleanup of a model's raw output before parsing it.

    A literate document IS markdown-with-fenced-code, so the standard
    ``sanitize_output`` is unusable here: it unwraps *any* leading ``` fence and
    would eat the first (and last) ```lackpy block of a doc that opens with code.
    This instead strips only a leading conversational preamble line and an
    explicit OUTER ``` ```markdown ``` / ``` ```md ``` wrapper -- never the inner
    ``` ```lackpy ``` fences the document is made of.
    """
    text = raw.strip()
    if not text:
        return ""
    lines = text.split("\n")
    while lines and any(m in lines[0].lower() for m in _LITERATE_PREAMBLE_MARKERS):
        lines = lines[1:]
    text = "\n".join(lines).strip()
    first = text.split("\n", 1)[0].strip().lower()
    if first in ("```markdown", "```md") and text.rstrip().endswith("```"):
        body = text.split("\n", 1)[1] if "\n" in text else ""
        text = body.rstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text


try:
    from kibitzer import KibitzerSession
    _HAS_KIBITZER = True
except ImportError:
    _HAS_KIBITZER = False


class LackpyService:
    """Unified service layer orchestrating the lackpy pipeline.

    Both the MCP server and CLI are thin adapters over this class.
    Initialize with a workspace path: programs run from it, and its
    ``PathPolicy`` governs their file access.

    When kibitzer is installed, the service automatically creates a
    KibitzerSession for mode enforcement, tool tracking, and coaching.

    Args:
        workspace: Root directory for tool execution. Defaults to cwd.
        config: Override configuration. Loaded from .lackpy/config.toml if not provided.
        path_policy_loader: Builds the ``PathPolicy`` from the workspace, once per
            run (settings may change between runs). Defaults to
            ``load_policy(workspace)``, which reads the managed settings, the
            user's ``~/.claude/settings.json`` and the workspace's settings.
            Tests pass a loader with a temporary home and ``managed=None``.
    """

    def __init__(self, workspace: Path | None = None, config: LackpyConfig | None = None,
                 harness_resolver: Any = None,
                 path_policy_loader: Callable[[Path], PathPolicy] | None = None) -> None:
        self._workspace = workspace or Path.cwd()
        self._config = config or load_config(self._workspace)
        # Optional host/harness resolver for virtual tools: name -> callable | None.
        self._harness_resolver = harness_resolver
        self._path_policy_loader = path_policy_loader
        # Plain programs run in a forked worker under a time limit (Plan 3 D9).
        # The runner is the process's shared one for the memory cap
        # (``shared_runner``), taken on the first run, so a service that never
        # runs starts no zygote. The service does not own it. The worker
        # changes into the workspace itself, so the service never calls
        # os.chdir and concurrent runs need no lock.
        self._process_runner: ProcessRunner | None = None
        # The live literate sessions a later run_program names by session_id
        # (Plan 3 Task 7, D10). Closed with the service.
        self._sessions = SessionRegistry()
        self._mcp_client: Any = None  # lazily created if [mcp_servers] are configured
        self.toolbox = Toolbox()
        # The toolbox is internal machinery since the kit was removed from the
        # runtime (stepy100 Task 7): no run binds its tools. It still feeds
        # kibitzer's tool registry and lackpyctl's toolbox listing.
        self.toolbox.register_provider(PythonProvider())
        for source, precedence in self._build_tool_sources():
            self.toolbox.add_source(source, precedence)
        self._inference_providers: list = []
        # The configured LLM model (set by _init_inference_providers from the woollama
        # tier) — recorded explicitly instead of reflecting `getattr(p, "_model")`.
        self._llm_model: str | None = None
        self._init_inference_providers()
        self._policy = PolicyLayer()
        self._policy.add_source(ToolsPolicySource(self.toolbox))
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
            tool_cfg = cfg.get("tools", {}) or {}
            overrides = {
                name: (g["w"], g["d"])
                for name, t in tool_cfg.items()
                if isinstance((g := (t or {}).get("grade")), dict) and "w" in g and "d" in g
            }
            # Few-shot examples per tool. Discovery yields a signature and a
            # description but nothing about usage, so a generator has to guess the
            # idiom; these reach the prompt through the same retrieval path as
            # builtin tools' examples rather than as prose in the intent.
            examples = {
                name: [e for e in ex if isinstance(e, dict) and e.get("intent") and e.get("code")]
                for name, t in tool_cfg.items()
                if isinstance((ex := (t or {}).get("examples")), list)
            }
            out.append((McpToolSource(spec, self._mcp_client, grade_overrides=overrides,
                                      example_overrides={k: v for k, v in examples.items() if v}),
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
        The run zygote is shared by every service in the process, so this
        drops the service's reference only; ``close_shared_runners`` (also
        run at interpreter exit) stops it.
        """
        if self._mcp_client is not None:
            self._mcp_client.shutdown()
            self._mcp_client = None
        self._sessions.close_all()
        self._process_runner = None

    def close_session(self, session_id: str) -> bool:
        """End the literate session ``session_id``; False when it is unknown or expired."""
        held = session_id in self._sessions
        self._sessions.close(session_id)
        return held

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

    def _resolve_profile(self, profile: "Profile | str | list[str] | dict | None") -> ResolvedProfile:
        """Resolve a profile to its model, mode, rules and execution axis.

        ``None`` is the configured default profile (``[profile] default``) when a
        ``[profiles.<name>]`` table defines it, and no profile otherwise. A
        ``str`` must name a ``[profiles.<name>]`` table. The resolved tool set is
        always empty: the kit was removed from the runtime (stepy100 Task 7).

        Raises:
            ProfileError: The profile names tools (a list, a dict, a ``.kit`` or
                ``.profile`` file, or a ``tools``/``extra_tools`` key), or the
                name is not configured; the message carries ``KIT_REMOVED``. Or
                the profile's ``policy.read_only`` is set and is not a bool.
        """
        profiles = self._config.profiles
        if profile is None:
            name = self._config.profile_default
            prof = self._configured_profile(name) if name in profiles else Profile()
        elif isinstance(profile, Profile):
            prof = profile
        elif isinstance(profile, str):
            if profile not in profiles:
                kits_dir = self._config.config_dir / "kits"
                files = [kits_dir / f"{profile}{ext}" for ext in (".profile", ".kit")]
                found = next((f for f in files if f.is_file()), None)
                if found is not None:
                    raise ProfileError(f"profile {profile!r} is a tool kit file ({found}): {KIT_REMOVED}")
                raise ProfileError(f"unknown profile {profile!r} (no [profiles.{profile}] table): "
                                   f"{KIT_REMOVED}")
            prof = self._configured_profile(profile)
        else:
            raise ProfileError(KIT_REMOVED)
        if prof.tools or prof.extra_tools:
            raise ProfileError(KIT_REMOVED)
        _read_only(prof.policy)  # a malformed read_only fails here, before any run
        _check_max_timeout(prof.max_timeout)
        _check_mode(prof.mode)
        return ResolvedProfile(
            tools=_no_tools(), model=prof.model, mode=prof.mode, order=prof.order,
            temperature=prof.temperature, rules=prof.rules, language=prof.language,
            execution=prof.execution, sandbox=prof.sandbox, policy=prof.policy,
            max_timeout=prof.max_timeout,
        )

    def _profile_name(self, profile: Any) -> str:
        """The profile's name for a message: ``(none)`` with no profile, and
        ``(unnamed)`` for a ``Profile`` passed as an object."""
        if isinstance(profile, str):
            return profile
        if profile is None:
            name = self._config.profile_default
            return name if name in self._config.profiles else "(none)"
        return "(unnamed)"

    def _check_timeout(self, timeout: Any, rp: ResolvedProfile,
                       profile: Any) -> tuple[float, str | None]:
        """The run's time limit in seconds, and the refusal text when it is refused (D8).

        The one place the rule lives. The cap is the profile's
        ``max_timeout``, else ``config.sandbox_timeout``. ``None`` is
        ``min(config.sandbox_timeout, cap)`` and is never refused (fix round 1
        ruling). An explicit value that is not a finite number above 0 is
        refused, and so is one over the cap.
        """
        cap = rp.max_timeout if rp.max_timeout is not None else self._config.sandbox_timeout
        if timeout is None:
            return min(self._config.sandbox_timeout, cap), None
        if not _positive_seconds(timeout):
            return 0.0, f"timeout must be a positive number of seconds, not {timeout!r}"
        seconds = timeout
        if seconds > cap:
            return seconds, (f"timeout {seconds:g} s is over the profile's max_timeout "
                             f"({cap:g} s, profile {self._profile_name(profile)})")
        return seconds, None

    def _configured_profile(self, name: str) -> Profile:
        try:
            return Profile.from_config(self._config.profiles[name])
        except ValueError as e:
            raise ProfileError(f"profile {name!r}: {e}") from None

    def _resolve_tool_selection(self, profile: Any, extra_tools: list[str] | None = None) -> ResolvedTools:
        """The retired kit resolution, for the internal toolbox views only
        (``profile_info``, ``docs_index``). No run uses it."""
        if profile is None:
            profile = self._config.profile_default
        kits_dir = self._config.config_dir / "kits"
        return resolve_profile(profile, self.toolbox, profiles=self._config.profiles,
                               kits_dir=kits_dir, extra_tools=extra_tools).tools

    def _resolve_params(self, params: dict[str, Any] | None) -> tuple[dict[str, Any], str | None]:
        """The passed values and the prompt's description of the params literal.

        Params bind no bare names (stepy100 Task 5), so no param name is ever
        allowed to the validator (Task 8)."""
        if not params:
            return {}, None
        values: dict[str, Any] = {}
        for name, val in params.items():
            # Only the single-key {"value": x} wrapper; any other dict is a dict param (review M7).
            if isinstance(val, dict) and val.keys() == {"value"}:
                values[name] = val["value"]
            else:
                values[name] = val
        # No name-collision check: params replace the values of the program's own
        # ``params`` literal and bind no bare names (stepy100 Task 5).
        return values, format_params_description(values)

    def validate(self, program: str, profile: Profile | str | None = None,
                 rules: list | None = None) -> ValidationResult:
        """Validate a lackpy program. No kit tool names are allowed.

        Args:
            program: The lackpy program source to validate. With a literate
                profile (``execution == "literate"``), a literate document: it is
                validated cell by cell, as a run validates it, and each
                violation names its document cell (final review M1).
            profile: A profile name or ``Profile``. A profile that names tools
                gives an invalid result carrying ``KIT_REMOVED``.
            rules: Additional validation rules to apply beyond core checks.

        Returns:
            A ValidationResult indicating whether the program is valid.
        """
        try:
            rp = self._resolve_profile(profile)
            rules = resolve_rules(rules)
        except (ProfileError, ValueError) as e:
            return ValidationResult(valid=False, errors=[str(e)])
        if rp.execution == "literate":
            from .interpreters.literate import validate_document
            return validate_document(program, extra_rules=rules)
        return validate(program, allowed_names=set(), extra_rules=rules)

    async def generate(self, intent: str, profile: Profile | str | None = None,
                       params: dict[str, Any] | None = None, rules: list | None = None,
                       mode: str | None = None, interpreter: Any = None,
                       model: str | None = None,
                       temperature: float | None = None) -> GenerationResult:
        """Generate a lackpy program from a natural language intent.

        Args:
            intent: Natural language description of the desired program.
            profile: A profile name or ``Profile`` (model, mode, rules).
            params: Named input values available to the generated program.
            rules: Additional validation rules the generated program must satisfy.
            mode: Inference strategy mode (e.g. '1-shot', 'spm'). Defaults to config or '1-shot'.
            interpreter: An interpreter instance whose ``system_prompt_hint()``
                is forwarded to the prompt builder: the intro of the lackpy
                prompt for the python interpreter, the whole framing for an
                interpreter of another language.
            model: Per-call LLM model override (``"<provider>/<model>"``). When set, the
                woollama tier is cloned to this model for this call only; when omitted,
                the service-configured model is used (identical to prior behavior). This
                is how a profile selects a per-task model (RFC 0002 incr. 5).
            temperature: Per-call temperature override for the LLM tier.

        Returns:
            A GenerationResult with the generated program and provider metadata.

        Raises:
            ProfileError: If the profile names tools (``KIT_REMOVED``) or is unknown.
            RuntimeError: If all inference providers fail to produce a valid program.
        """
        from .infer.strategy import STRATEGIES
        from .infer.context import StepContext

        rp = self._resolve_profile(profile)
        resolved = rp.tools
        rules = resolve_rules(rules)
        _, params_desc = self._resolve_params(params)

        # Inference selections: an explicit call arg wins over the profile's; both fall
        # back to the service config (woollama tier / inference_mode) downstream.
        effective_model = model if model is not None else rp.model
        effective_temp = temperature if temperature is not None else rp.temperature
        # Per-call providers: the global set, or a copy with the LLM tier overridden when a
        # model/temperature is selected. Neither selected → identical to before (invariant 8).
        providers = self._providers_for(effective_model, effective_temp)

        # Literate execution needs a literate DOCUMENT, not restricted Python, so it
        # gets its own generation path (own hint + parse validation) rather than the
        # strategy/dispatcher paths, which frame + validate output as Python.
        if rp.execution == "literate":
            return await self._generate_literate(intent, providers, resolved, params_desc, rules)

        effective_mode = mode or rp.mode or self._config.inference_mode
        _check_mode(effective_mode)

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
                intent=intent, tools=resolved,
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
        # No kit tools and no param names: params bind no bare names (Task 5).
        allowed: set[str] = set()

        # Resolve policy — kit baseline + optional kibitzer hints + optional umwelt.
        # The active model is known explicitly (per-call override or the configured LLM
        # model) — no more reflecting `getattr(provider, "_model")`.
        from .policy.types import PolicyContext, ModelSpec
        model_name = effective_model or self._llm_model
        policy_context: PolicyContext = {"tools": resolved}
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

    async def _execute(self, program: str, param_values: dict[str, Any], *,
                       timeout: float, read_only: bool = False,
                       rules: list[str] | None = None) -> ExecutionResult:
        """Run an (already validated) program in a worker, under ``timeout`` seconds.

        The single execution path shared by ``run_program`` and ``delegate``
        (Plan 3 D9). The PathPolicy is loaded here, in the service, on every
        run (a settings change takes effect on the next run), and crosses to
        the worker as data. A malformed settings file gives a failed result
        whose error names the file; it never raises. The worker changes into
        the workspace and validates again, with the ``rules`` names.
        ``ProcessRunner.run`` blocks, so it runs in a thread; the zygote is a
        separate process, so the service's threads do not reach it.
        """
        try:
            policy = self._load_path_policy(read_only=read_only)
        except (Exception, LackpyRefusal) as e:  # host code; never raises (review M4, I4)
            return ExecutionResult(success=False, error=_policy_error(e))
        if self._process_runner is None:
            self._process_runner = shared_runner(self._config.sandbox_memory_mb)
        runner = self._process_runner
        try:
            return await asyncio.to_thread(
                runner.run, program, params=param_values, path_policy=policy,
                workspace=self._workspace.resolve(), timeout=timeout, rules=rules)
        except RuntimeError as e:  # no os.fork: loud, and never an in-process run
            return ExecutionResult(success=False, error=str(e))

    def _load_path_policy(self, *, read_only: bool) -> PathPolicy:
        """The workspace ``PathPolicy`` through the service's policy seam, loaded
        for this run; with ``read_only`` it also denies every write
        (``read_only_policy``, design D7, Ruling F). Raises what the loader raises."""
        loader = self._path_policy_loader or load_policy
        policy = loader(self._workspace)
        return read_only_policy(policy) if read_only else policy

    async def _execute_program(
        self, program: str, rp: ResolvedProfile, param_values: dict[str, Any],
        *, timeout: float | None = None, rules: list | None = None,
        rule_names: list[str] | None = None, session: tuple[str | None, str] | None = None,
    ) -> ExecutionResult:
        """Route a program to the execution model named by the profile's ``execution``
        axis -- the shared seam for ``run_program`` and ``delegate``.

        Validation lives ABOVE the branch: a literate document is NOT restricted
        Python, so it must not pass through the restricted-Python validator (which
        would reject its fences/prose as a syntax error and never reach the gate).
        ``execution == "literate"`` runs the document through the
        LiterateInterpreter: every cell is validated against the program so far
        and runs in one CellSession under the workspace PathPolicy (literate
        stepy D2, D9). Any other axis is restricted Python -- always validated
        with no allowed names (parked item 1: there is no skip), with the
        caller's ``rules`` (callables included), then run in a worker under
        ``timeout`` seconds (Plan 3 D9), which validates again with
        ``rule_names``. The literate axis runs in a session worker, with the
        limit on each cell, and validates there with ``rule_names`` only (a
        callable rule is refused: it cannot cross; ledger Ruling D). A profile whose ``policy.read_only`` is
        true runs under ``read_only_policy`` on either axis.

        ``session`` is ``(session_id, profile_name)`` from ``run_program`` (Plan 3
        Task 7, D10): a literate run then stays open in the registry, and a
        given ``session_id`` resumes it. None runs and closes, as before.
        """
        read_only = _read_only(rp.policy)
        if timeout is None:  # an internal caller: the same capped default (review Minor 3)
            timeout, _ = self._check_timeout(None, rp, None)
        if rp.execution == "literate":
            # ``rules`` is the resolved list (one callable per rule); a rule the
            # caller passed as a callable has no name in ``rule_names``.
            if len(rules or []) != len(rule_names or []):
                return ExecutionResult(
                    success=False,
                    error="the process runner takes rule names, not callables")
            return await self._execute_literate(program, param_values, timeout=timeout,
                                                read_only=read_only, rule_names=rule_names,
                                                session=session)
        if session is not None and session[0] is not None:
            return ExecutionResult(success=False,
                                   error="session_id applies only to a literate profile")

        validation = validate(program, allowed_names=set(), extra_rules=rules)
        if not validation.valid:
            return ExecutionResult(success=False, error=validation_error(validation))
        return await self._execute(program, param_values, timeout=timeout,
                                   read_only=read_only, rules=rule_names)

    async def _execute_literate(
        self, program: str, param_values: dict[str, Any], *, timeout: float,
        read_only: bool = False, rule_names: list[str] | None = None,
        session: tuple[str | None, str] | None = None,
    ) -> ExecutionResult:
        """Run a literate document in a session worker (Plan 3 D6, Ruling D) and adapt the result.

        The ``PathPolicy`` comes through the same seam as ``run_program``'s
        (``_load_path_policy``); a load failure is a failed result, never a
        raise. It crosses to the worker as data. The whole literate interpreter
        runs in the worker, with a checkpoint fork at the top of each cell, and
        each cell runs under ``timeout`` seconds: a cell over the limit is
        killed, reified as timed out, and the document goes on (D7,
        ``error_phase="timeout"``). The document is parsed and every cell is
        validated in the worker, with the caller's rules as names
        (``rule_names``).
        No toolset is passed: there is no kit grade, so no static ceiling; the
        policy is the boundary (a read-only profile adds the write deny).

        The result: ``stdout`` is the rendered document; ``output`` follows
        ``run_program``'s last-expression rule: when the run succeeded and the
        last code cell ends in an expression statement, that expression's
        value (data, or its ``repr`` text), else ``None``; ``trace`` is what
        the cells streamed, a stopped cell's entries included; ``error`` is the
        refusal that ended the document, then the reified failures. A refusal
        outranks a timeout in ``error_phase``. A read-only profile also passes a
        read-only ``grade_ceiling``, so the static planner refuses a visible
        write cell before it runs (the policy still refuses what the planner
        cannot see). ``param_values`` replace the defaults of the document's
        ``params`` literal (final review I4, D15).
        """
        try:
            policy = self._load_path_policy(read_only=read_only)
        except (Exception, LackpyRefusal) as e:  # host code; never raises (review M4, I4)
            return ExecutionResult(success=False, error=_policy_error(e))
        config: dict[str, Any] = {}
        if read_only:
            # Task 4 review M5: the static planner over the policy (ruling) —
            # a write cell it can see is refused before it runs.
            config["grade_ceiling"] = _READ_ONLY_CEILING
            # Final review M5: the refusal says why the ceiling is there.
            config["grade_ceiling_reason"] = "the profile is read-only (policy.read_only = true)"
        if self._process_runner is None:
            self._process_runner = shared_runner(self._config.sandbox_memory_mb)
        runner = self._process_runner
        workspace = self._workspace.resolve()

        def run_in_session(session_id: str | None, profile_name: str) -> ExecutionResult:
            # Task 7 (D10): the session stays open for a retry, and a retry
            # with its session_id resumes from the first changed cell.
            if session_id is None:
                live = self._sessions.open(runner=runner, workspace=workspace,
                                           profile_name=profile_name, path_policy=policy,
                                           cell_timeout=timeout, rule_names=rule_names,
                                           context_config=config)
            else:
                try:
                    live = self._sessions.get(session_id, workspace=workspace,
                                              profile_name=profile_name)
                except SessionError as e:
                    return ExecutionResult(success=False, error=str(e))
                live.configure(path_policy=policy, cell_timeout=timeout,
                               rule_names=rule_names, context_config=config)
            result = live.submit(program, param_values)
            result.session_id = live.session_id
            return result

        def run() -> ExecutionResult:
            if session is not None:
                return run_in_session(*session)
            channel = runner.open_session(path_policy=policy, workspace=workspace,
                                          cell_timeout=timeout, rule_names=rule_names,
                                          context_config=config)
            try:
                return channel.submit(program, params=param_values)
            finally:
                channel.close()

        try:
            return await asyncio.to_thread(run)
        except (RuntimeError, ValueError) as e:  # no os.fork, no session, a callable rule
            return ExecutionResult(success=False, error=str(e))

    async def _generate_literate(
        self, intent: str, providers: list, resolved: Any,
        params_desc: str | None, rules: list | None,
    ) -> "GenerationResult":
        """Generate a literate document (the ``execution == "literate"`` path).

        A literate doc is markdown, not restricted Python, so it CANNOT go through
        the standard dispatcher: that path runs ``sanitize_output`` +
        ``deterministic_cleanup`` + validates the result AS Python + a Python
        correction chain -- all of which reject or mangle a fenced document. This
        runs the provider loop with the LiterateInterpreter system-prompt hint (the
        stepy lesson, the literate format, one worked document) and
        validates each candidate by PARSING it (``LiterateInterpreter.validate``),
        returning the first that parses. Cleanup is literate-safe (outer wrapper /
        preamble only). No Python sanitize/correction.
        """
        from .infer.dispatch import GenerationResult
        from .interpreters.base import ExecutionContext
        from .interpreters.literate import LiterateInterpreter

        interp = LiterateInterpreter()
        vctx = ExecutionContext(base_dir=str(self._workspace.resolve()), tools=resolved)
        # No tool is taught (literate stepy D10): the prompt is the stepy lesson, the
        # literate format and one document, so no namespace description reaches it.
        namespace_desc = ""
        start = time.perf_counter()
        # With params, the prompt also teaches the document's params literal (as
        # the one-shot prompt does), so the run can replace its values (D15).
        system = None
        if params_desc:
            from .infer.prompt import build_system_prompt
            from .interpreters.literate.prompt import params_section
            system = (build_system_prompt(namespace_desc, interpreter=interp)
                      + params_section(params_desc))
        errors: dict[str, list[str]] = {}
        for provider in providers:
            if not provider.available():
                continue
            raw = await provider.generate(intent, namespace_desc, interpreter=interp,
                                          system_prompt_override=system)
            if not raw:
                continue
            program = _strip_literate_wrapper(raw)
            validation = interp.validate(program, vctx)
            if validation.valid:
                elapsed = (time.perf_counter() - start) * 1000
                return GenerationResult(
                    program=program, provider_name=provider.name,
                    generation_time_ms=elapsed,
                )
            errors[provider.name] = validation.errors
        raise RuntimeError(f"Literate generation failed. Errors: {errors}")

    def _planned_tool_calls(self, program: str, rp: ResolvedProfile) -> list[str]:
        """Tool names a generated program will call, for the kibitzer mode-policy
        pre-check (Python builtins excluded).

        A literate document is markdown, so the restricted-Python validator rejects
        it outright and the pre-check used to silently self-skip -- mode policy did
        not apply to literate calls. Compile the doc to its Python form first, then
        extract calls the same way (``validate`` collects called names during its
        AST walk regardless of whether other rules pass, so an off-lattice doc's
        tool calls are still surfaced). One-shot programs are unchanged: calls are
        taken only when the program validates. Returns ``[]`` when nothing can be
        analyzed (execution will surface the real error). Since the kit was
        removed no name is allowed, so a valid one-shot program calls only
        builtins, imports and its own defs, and gives ``[]``.

        COOPERATIVE, not a security boundary -- the same limitation as the effect
        gate. This enumerates *direct, named* calls only. An attribute call
        (``os.system(...)``), an alias (``w = write_file; w(...)``), or reflection
        (``getattr(...)(...)``) is NOT attributed to a tool and slips the check.
        Soundness against adversarial docs comes from the restricted-AST whitelist
        + nsjail, not from this pass.
        """
        if rp.execution == "literate":
            try:
                from .interpreters.literate.compiler import (
                    LITERATE_RUNTIME_INTERNALS,
                    compile_document,
                )
                program = compile_document(program)
            except Exception:
                return []
            # The compiler emits runtime-internal calls (the @continue
            # sentinel, the prose display helper) that are plumbing, not
            # writer-planned tool calls -- filter them so mode policy doesn't
            # see phantom tools.
            calls = [
                c for c in validate(program, allowed_names=set()).calls
                if c not in LITERATE_RUNTIME_INTERNALS
            ]
        else:
            vr = validate(program, allowed_names=set())
            if not vr.valid:
                return []
            calls = vr.calls
        return [c for c in calls if c not in ALLOWED_BUILTINS]

    async def run_program(self, program: str, profile: Profile | str | None = None,
                          params: dict[str, Any] | None = None, sandbox: Any = None,
                          rules: list | None = None,
                          timeout: float | None = None,
                          session_id: str | None = None) -> ExecutionResult:
        """Validate and execute a lackpy program.

        The program runs in the runtime namespace (builtins, approved modules,
        the ``Path`` stand-in) under ``load_policy(workspace)``, loaded for
        this run. It is validated with no allowed names first. Every failure,
        including a profile that names tools and a malformed settings file,
        is a failed ExecutionResult, never a raise.

        Args:
            program: The lackpy program source to execute.
            profile: A profile name or ``Profile`` (model, mode, rules, execution).
            params: Values for the program's ``params`` literal.
            sandbox: Reserved for future sandbox configuration (unused).
            rules: Additional validation rules to apply before execution.
            timeout: The run's time limit in seconds (Plan 3 D8); None is
                ``config.sandbox_timeout``. Over the profile's ``max_timeout``
                (default ``config.sandbox_timeout``), or not a number above 0,
                the run is refused before validation (``error_phase="refused"``).
                At the limit the worker is killed (``error_phase="timeout"``).
                In a literate document the limit applies to each cell (D8).
            session_id: A literate run's ``session_id`` (Plan 3 D10). Every
                literate run returns one in ``result.session_id``; passing it
                back with a resubmitted document resumes that session from the
                checkpoint before the first changed cell. An unknown or expired
                id, or one from another profile or workspace, is a failed
                result, never a fresh run.

        Returns:
            An ExecutionResult with output, trace, and success status.
        """
        try:
            rp = self._resolve_profile(profile)
            names = _rule_names(rules)
            rules = resolve_rules(rules)
        except (ProfileError, ValueError) as e:
            return ExecutionResult(success=False, error=str(e))
        seconds, refusal = self._check_timeout(timeout, rp, profile)
        if refusal is not None:
            return ExecutionResult(success=False, error=refusal, error_phase="refused")
        param_values, _ = self._resolve_params(params)
        # The execution axis (one-shot restricted-Python vs literate) decides both
        # validation and execution; the seam keeps them paired (see _execute_program).
        return await self._execute_program(program, rp, param_values, timeout=seconds,
                                           rules=rules, rule_names=names,
                                           session=(session_id, self._profile_name(profile)))

    async def delegate(self, intent: str, profile: Profile | str | None = None,
                       params: dict[str, Any] | None = None, sandbox: Any = None,
                       rules: list | None = None,
                       _program_override: str | None = None,
                       mode: str | None = None,
                       interpreter: Any = None,
                       model: str | None = None,
                       temperature: float | None = None,
                       timeout: float | None = None) -> dict[str, Any]:
        """Generate and execute a program from a natural language intent in one step.

        Combines generate and run_program: generates a program from intent, then
        executes it, returning a combined result dict with timing and trace details.

        Args:
            intent: Natural language description of the desired program.
            profile: A profile name or ``Profile`` (model, mode, rules). A profile
                that names tools gives a failed result carrying ``KIT_REMOVED``.
            params: Named input values available to the program.
            sandbox: Reserved for future sandbox configuration (unused).
            rules: Additional validation rules for generation and execution.
            interpreter: An interpreter instance whose ``system_prompt_hint()``
                is forwarded to the prompt builder. When omitted, the default
                lackpy prompt is used (pre-prompt, style guide, skeleton).
            timeout: The run's time limit in seconds, as ``run_program``'s. A
                refused value is refused before generation. A timed-out run is
                returned as it is: the caller decides on a retry (D7).

        Returns:
            A dict with keys: success, program, generation_tier,
            generation_time_ms, execution_time_ms, total_time_ms, trace,
            files_read, files_modified, output, error, error_phase.

        Raises:
            RuntimeError: If all inference providers fail to produce a valid program.
        """
        start = time.perf_counter()
        try:
            rp = self._resolve_profile(profile)
            if not _program_override:  # only generation runs the strategy
                _check_mode(mode or rp.mode or self._config.inference_mode)
            names = _rule_names(rules)
            rules = resolve_rules(rules)
        except (ProfileError, ValueError) as e:
            return {"success": False, "program": _program_override, "output": None,
                    "stdout": "", "error": str(e), "error_phase": None, "trace": []}
        seconds, refusal = self._check_timeout(timeout, rp, profile)
        if refusal is not None:  # before generation: no model call for a refused run
            return {"success": False, "program": _program_override, "output": None,
                    "stdout": "", "error": refusal, "error_phase": "refused", "trace": []}
        # The model actually used (per-call override or the profile's, else configured) —
        # the same value `generate` uses, so kibitzer logs the real model, not a reflection.
        effective_model = (model if model is not None else rp.model) or self._llm_model
        param_values, _ = self._resolve_params(params)

        # Kibitzer: register context for this delegation
        if self._kibitzer:
            self._kibitzer.register_context({
                "source": "lackpy",
                "intent": intent,
                "tools": [],
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
                                            interpreter=interpreter,
                                            model=model, temperature=temperature)

        # Kibitzer: validate planned calls before execution. A literate document is
        # compiled to its Python form first (see _planned_tool_calls) so mode policy
        # applies to literate calls too, instead of self-skipping on the markdown.
        kibitzer_suggestions: list[str] = []
        if self._kibitzer:
            planned_names = self._planned_tool_calls(gen_result.program, rp)
            if planned_names:  # only ever validate a non-empty call set
                planned = [{"tool": call, "input": {}} for call in planned_names]
                violations = self._kibitzer.validate_calls(planned)
                if violations:
                    return {
                        "success": False,
                        "program": gen_result.program,
                        "generation_tier": gen_result.provider_name,
                        "error": "Kibitzer mode policy violation: " + "; ".join(v.reason for v in violations),
                        "error_phase": None,
                        # The keys every other delegate result has (review M4).
                        "output": None, "stdout": "", "trace": [],
                        "correction_strategy": gen_result.correction_strategy,
                        "correction_attempts": gen_result.correction_attempts,
                    }

        # Dispatch by the profile's execution axis (literate vs restricted Python),
        # the same seam run_program uses. Defense-in-depth (GHSA-hpcj-3c97-43jm):
        # re-validate the generated program (with no allowed names) at the
        # execution boundary, rather than trusting the generation-time gate alone.
        # The restricted-Python path parses the identical source deterministically,
        # so this closes the validate/execute gap without a differential; the
        # literate path validates internally by parsing.
        # Review M5: the caller's rules are re-applied here too.
        exec_result = await self._execute_program(gen_result.program, rp, param_values,
                                                  timeout=seconds, rules=rules, rule_names=names)

        # Kibitzer: get coaching suggestions after execution
        if self._kibitzer:
            kibitzer_suggestions = self._kibitzer.get_suggestions()
            # Classify failure mode for kibitzer's pattern tracker
            failure_mode = None
            if not exec_result.success:
                from .infer.failure_modes import classify_failure
                validation_result = validate(gen_result.program, allowed_names=set())
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
            "generation_tier": gen_result.provider_name,
            "generation_time_ms": gen_result.generation_time_ms,
            "execution_time_ms": total_ms - gen_result.generation_time_ms,
            "total_time_ms": total_ms,
            "trace": [{"step": e.step, "tool": e.tool, "args": e.args, "result": e.result,
                        "duration_ms": e.duration_ms, "success": e.success, "error": e.error}
                       for e in exec_result.trace.entries],
            # Review M3: from the Path stand-in's successful trace entries.
            "files_read": _traced_paths(exec_result.trace, ("Path.read_text", "Path.read_bytes")),
            "files_modified": _traced_paths(exec_result.trace,
                                            ("Path.write_text", "Path.write_bytes")),
            # Prefer the typed last-expression value; if the program instead
            # printed its answer (a common small-model habit), fall back to the
            # captured stdout so the result isn't silently dropped. stdout is
            # always surfaced so the caller can see it regardless.
            "output": exec_result.effective_output,
            "stdout": exec_result.stdout,
            "error": exec_result.error,
            "error_phase": exec_result.error_phase,
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
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Load and run a Lackey file (``timeout`` as ``delegate``'s).

        The run body reads each param as a bare name (``self.path`` becomes
        ``path``), and params bind no bare names since stepy100 Task 5. So the
        body runs behind a prepended ``params = {...}`` literal of the
        Lackey's defaults and one ``name = params["name"]`` binding per param
        (parked item 4). Passed ``params`` replace the literal's values, with
        the runner's type check. A param with no default holds ``None`` in the
        literal and must be passed. A Lackey that declares ``Tool()``
        attributes gives a failed result carrying ``KIT_REMOVED``.
        """
        from .lackey.parser import parse_lackey as _parse

        info = _parse(path)
        if info.tools:
            return {"success": False, "program": info.run_body, "output": None,
                    "stdout": "", "error": KIT_REMOVED, "trace": []}
        passed = dict(params or {})
        missing = [n for n, spec in info.params.items() if "default" not in spec and n not in passed]
        if missing:
            return {"success": False, "program": info.run_body, "output": None, "stdout": "",
                    "error": f"Lackey param {missing[0]!r} has no default and was not passed",
                    "trace": []}

        defaults = {n: spec.get("default") for n, spec in info.params.items()}
        # Convert top-level `return X` to bare expression `X` so the runner can
        # execute the body as a flat program (return is only valid inside a function).
        run_body = _strip_top_level_return(info.run_body)
        # Review M5: the prefix below owns the name `params`.
        if "params" in defaults:
            return {"success": False, "program": info.run_body, "output": None, "stdout": "",
                    "error": "Lackey param 'params' collides with the params literal that "
                             "run_lackey adds; rename the param", "trace": []}
        import ast as _ast
        if defaults and any(_binds_params(n) for n in _ast.walk(_ast.parse(run_body))):
            return {"success": False, "program": info.run_body, "output": None, "stdout": "",
                    "error": "Lackey run body binds 'params', but run_lackey adds the params "
                             "literal for the Lackey's params; rename the body's variable",
                    "trace": []}
        prefix = ""
        if defaults:
            prefix = f"params = {defaults!r}\n" + "".join(
                f"{n} = params[{n!r}]\n" for n in defaults)

        return await self.delegate(
            intent="",
            params=passed or None,
            _program_override=prefix + run_body,
            timeout=timeout,
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

    async def create(self, program: str, profile: Profile | str | None = None,
                     name: str = "", pattern: str | None = None) -> dict[str, Any]:
        """Validate a program and save it as a named template.

        Args:
            program: The lackpy program source to save.
            profile: A profile name or ``Profile``. A profile that names tools
                gives ``{"success": False, "errors": [KIT_REMOVED]}``.
            name: Template name (used as the filename stem).
            pattern: Optional intent pattern string for template matching.

        Returns:
            A dict with keys: success (bool), path (str) on success,
            or errors (list) on validation failure.
        """
        try:
            self._resolve_profile(profile)
        except ProfileError as e:
            return {"success": False, "errors": [str(e)]}
        # Review M7: the name is a file name stem, never a path.
        if any(c in name for c in ("/", "\\", "\x00", "..")) or name == ".":
            return {"success": False,
                    "errors": [f"template name {name!r} must be a plain file name: "
                               "no '/', '\\', NUL, '.' or '..'"]}
        validation = validate(program, allowed_names=set())
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
        resolved = self._resolve_tool_selection(profile, extra_tools)
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
        resolved = self._resolve_tool_selection(profile, extra_tools)
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
