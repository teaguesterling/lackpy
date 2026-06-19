"""Profiles — named per-task config bundles (RFC 0002 increment 5).

A *profile* generalizes a "kit": instead of selecting only tools, it bundles, per task, a
tool selection + inference config (model/mode/…) + which language/execution model +
optional policy defaults. A **kit is the degenerate profile** that sets only ``tools``.

Thin by design (maintainer's decision): a ``Profile`` is a bundle of *references*, and
resolving it **composes existing machinery** — the tool selection goes through the
unchanged ``resolve_kit`` (so grade/policy/validation are untouched), and the rest is
carried alongside. Resolution yields a ``ResolvedProfile`` that is **driver-agnostic**
(one-shot, incremental, conversation, eventful, … all consume it — RFC §3.6) and never
assumes a single synchronous call; results flow through the emit seam
(:mod:`lackpy.run.events`), not an assumed return value.

Phase 1 is purely additive: this does not touch the existing kit/service call sites.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .kit.registry import ResolvedKit, resolve_kit
from .kit.toolbox import Toolbox
from .lang.grader import Grade

DEFAULT_LANGUAGE = "restricted-python"
DEFAULT_EXECUTION = "one-shot"

# Recognized [profiles.<name>] keys — anything else is rejected so typos surface early.
_PROFILE_KEYS = frozenset({
    "tools", "extra_tools", "model", "mode", "order", "temperature",
    "rules", "language", "execution", "sandbox", "policy",
})


@dataclass(frozen=True)
class Profile:
    """A named per-task config bundle (the *input*). Thin: references, not resolved state.

    Every field is optional; a profile that sets only ``tools`` is the degenerate "kit".
    Inference fields left ``None`` fall back to the service config at resolve time
    (so an unset profile costs nothing — invariant 8). ``language``/``execution`` are the
    two interpreter axes (RFC §3.5); their defaults are today's only supported path.
    """

    tools: str | list[str] | dict[str, Any] | None = None
    extra_tools: list[str] | None = None
    # inference (service-global today → per-profile here)
    model: str | None = None
    mode: str | None = None
    order: list[str] | None = None
    temperature: float | None = None
    rules: list[str] | None = None
    # interpreter axes (RFC §3.5): language profile × execution model
    language: str = DEFAULT_LANGUAGE
    execution: str = DEFAULT_EXECUTION
    sandbox: str | None = None
    policy: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "Profile":
        """Build a Profile from a ``[profiles.<name>]`` config table (raw dict)."""
        unknown = set(data) - _PROFILE_KEYS
        if unknown:
            raise ValueError(
                f"unknown profile key(s): {sorted(unknown)}; allowed: {sorted(_PROFILE_KEYS)}"
            )
        return cls(**{k: data[k] for k in _PROFILE_KEYS if k in data})


@dataclass
class ResolvedProfile:
    """A resolved profile — a driver-agnostic, session-able unit.

    Holds the resolved tools (the existing ``ResolvedKit``, with its tool-derived grade)
    plus the carried inference/language/execution/policy selections. Drivers consume this;
    nothing here assumes a single synchronous call (invariant 9).
    """

    tools: ResolvedKit
    model: str | None = None
    mode: str | None = None
    order: list[str] | None = None
    temperature: float | None = None
    rules: list[str] | None = None
    language: str = DEFAULT_LANGUAGE
    execution: str = DEFAULT_EXECUTION
    sandbox: str | None = None
    policy: dict[str, Any] | None = None

    # A profile's grade/description ARE its tools' — never independently set (invariant 3).
    @property
    def grade(self) -> Grade:
        return self.tools.grade

    @property
    def description(self) -> str:
        return self.tools.description


def resolve_profile(
    profile: "Profile | str | list[str] | dict[str, Any] | None",
    toolbox: Toolbox,
    *,
    profiles: dict[str, dict[str, Any]] | None = None,
    kits_dir: Path | None = None,
    defaults: dict[str, Any] | None = None,
    extra_tools: list[str] | None = None,
) -> ResolvedProfile:
    """Resolve a profile into a ``ResolvedProfile``, composing the unchanged ``resolve_kit``.

    ``profile`` may be:

    - a ``Profile`` — used as-is;
    - a ``str`` naming a ``[profiles.<name>]`` entry in ``profiles`` — expanded;
    - any tool-selection value (``str`` tool/tool-set name, ``list``, ``dict``, ``None``) —
      treated as the **degenerate, tools-only profile** ("a kit is a profile").

    Inference fields the profile leaves unset fall back to ``defaults`` (the service config).
    ``extra_tools`` is a runtime addition merged on top of the profile's own ``extra_tools``
    (the harness/CLI adding tools beyond the profile's declared set). Tool resolution, grade,
    policy, and validation go through the existing machinery untouched.
    """
    prof = _coerce(profile, profiles)
    merged_extra = (prof.extra_tools or []) + (extra_tools or []) or None
    resolved_tools = resolve_kit(
        prof.tools, toolbox, kits_dir=kits_dir, extra_tools=merged_extra
    )
    d = defaults or {}

    def _or_default(value: Any, key: str) -> Any:
        return value if value is not None else d.get(key)

    return ResolvedProfile(
        tools=resolved_tools,
        model=_or_default(prof.model, "model"),
        mode=_or_default(prof.mode, "mode"),
        order=_or_default(prof.order, "order"),
        temperature=_or_default(prof.temperature, "temperature"),
        rules=prof.rules,
        language=prof.language,
        execution=prof.execution,
        sandbox=_or_default(prof.sandbox, "sandbox"),
        policy=prof.policy,
    )


def _coerce(
    profile: "Profile | str | list[str] | dict[str, Any] | None",
    profiles: dict[str, dict[str, Any]] | None,
) -> Profile:
    """Normalize the ``profile`` argument to a ``Profile``.

    A ``str`` that names a configured profile expands to it; any other value is a bare
    tool selection → the degenerate, tools-only profile.
    """
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, str) and profiles and profile in profiles:
        return Profile.from_config(profiles[profile])
    return Profile(tools=profile)
