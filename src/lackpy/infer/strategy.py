"""Inference strategies: named fold configurations over steps and combinators."""

from __future__ import annotations

from typing import Any, Protocol

from .combinators import Fallback, Sequence
from .steps.generate import GenerateStep
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


STRATEGIES: dict[str, type] = {
    "1-shot": OneShotStrategy,
    "spm": SPMStrategy,
}
