"""Inference strategies: named fold configurations over steps and combinators."""

from __future__ import annotations

from typing import Any, Protocol

from .combinators import Fallback, RetryWithFeedback, Sequence
from .steps.generate import GenerateStep
from .steps.generate_dsl import GenerateDSLStep
from .steps.interpret import InterpretStep
from .steps.validate import ValidateStep
from .steps.cleanup import CleanupStep
from .steps.few_shot import FewShotCorrectStep
from .steps.fresh_fix import FreshFixStep
from .steps.solve import SolveStep
from .steps.pick import PickStep
from .steps.restrict import RestrictStep


class InferenceStrategy(Protocol):
    """A named composition of steps that produces a valid lackpy program."""

    name: str

    def build(self, provider: Any) -> Any: ...


class OneShotStrategy:
    """Current behavior: generate -> validate -> correct on failure.

    Fallback chain: validate as-is, cleanup, few-shot correct, fresh fix.
    """

    name = "1-shot"

    def build(self, provider: Any) -> Any:
        return Sequence([
            GenerateStep(provider),
            Fallback([
                Sequence([ValidateStep()]),
                Sequence([CleanupStep(), ValidateStep()]),
                Sequence([FewShotCorrectStep(provider), ValidateStep()]),
                Sequence([FreshFixStep(provider), ValidateStep()]),
            ]),
        ])


class SPMStrategy:
    """Solve-Pick-Restrict: separate problem solving from constraint compliance.

    1. Solve: generate unconstrained standard Python
    2. Pick: analyze AST to determine which tools are needed
    3. Restrict: rewrite into lackpy subset (with fallback correction)
    """

    name = "spm"

    def build(self, provider: Any) -> Any:
        return Sequence([
            SolveStep(provider),
            PickStep(),
            Fallback([
                Sequence([RestrictStep(provider), ValidateStep()]),
                Sequence([RestrictStep(provider), CleanupStep(), ValidateStep()]),
                Sequence([RestrictStep(provider), FewShotCorrectStep(provider), ValidateStep()]),
            ]),
        ])


class StagedDSLStrategy:
    """Staged interpretation: generate DSL program → execute via interpreter.

    intent → generate(selector via interpreter hint) → execute(selector via pluckit)

    Each stage has a constrained output space (a DSL), which is both
    machine-verifiable and embeddable as input to the next stage. This
    gives small models (coder:3b) 87% first-try accuracy on 30 selector
    tasks with 100% execution rates.

    Uses RetryWithFeedback to wrap generation with interpreter validation
    and error-guided retry before execution.

    When ``execute_in_retry=True``, InterpretStep runs as a Check inside
    the retry loop — execution failures feed back into the next generation
    attempt. Use this for DSLs where runtime failures are common (SQL).
    When False (default), execution runs once after the retry loop — suited
    for DSLs with near-100% execution rates (CSS selectors).
    """

    name = "staged-dsl"

    def __init__(self, interpreter: Any = None, config: dict | None = None,
                 base_dir: str = ".", max_retries: int = 2,
                 checks: list | None = None,
                 execute_in_retry: bool = False) -> None:
        self._interpreter = interpreter
        self._config = config or {}
        self._base_dir = base_dir
        self._max_retries = max_retries
        self._extra_checks = checks or []
        self._execute_in_retry = execute_in_retry

    def build(self, provider: Any) -> Any:
        interpreter = self._interpreter
        if interpreter is None:
            from ..interpreters.ast_select import AstSelectInterpreter
            interpreter = AstSelectInterpreter()

        interpret = InterpretStep(interpreter, config=self._config, base_dir=self._base_dir)
        all_checks = [interpreter] + self._extra_checks

        if self._execute_in_retry:
            all_checks = all_checks + [interpret]

        return Sequence([
            RetryWithFeedback(
                GenerateDSLStep(provider, interpreter),
                checks=all_checks,
                max_retries=self._max_retries,
            ),
            interpret,
        ])


STRATEGIES: dict[str, type] = {
    "1-shot": OneShotStrategy,
    "spm": SPMStrategy,
    "staged-dsl": StagedDSLStrategy,
}
