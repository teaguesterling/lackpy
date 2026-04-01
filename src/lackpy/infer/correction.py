"""Error correction chain: deterministic cleanup, few-shot retry, fresh fixer."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..lang.validator import validate
from .cleanup import deterministic_cleanup
from .fixer import build_fixer_messages
from .hints import enrich_errors
from .sanitize import sanitize_output


@dataclass
class CorrectionAttempt:
    """Record of a single correction attempt.

    Attributes:
        strategy: The strategy used: "deterministic_cleanup", "few_shot_correction",
            or "fresh_fixer".
        program: The program produced by this attempt.
        errors: Validation errors on this program (empty if accepted).
        accepted: Whether this attempt produced a valid program.
    """

    strategy: str
    program: str
    errors: list[str]
    accepted: bool


@dataclass
class CorrectionResult:
    """Result of a successful correction.

    Attributes:
        program: The corrected, valid program.
        strategy: The strategy that produced the valid program.
        attempts: Total number of correction attempts made.
    """

    program: str
    strategy: str
    attempts: int


async def _call_fixer(provider, messages: list[dict]) -> str | None:
    """Call the provider's low-level chat/message API with fixer messages.

    Tries ``provider._chat(messages, temperature=0.4)`` first, then falls back
    to ``provider._create_message(system, user_messages)``.

    Args:
        provider: Inference provider with ``_chat`` or ``_create_message``.
        messages: List of message dicts as returned by ``build_fixer_messages()``.

    Returns:
        Raw response text, or None if all attempts fail or raise.
    """
    if hasattr(provider, "_chat"):
        try:
            response = await provider._chat(messages, temperature=0.4)
            content = response.get("message", {}).get("content", "")
            return content if content else None
        except Exception:
            pass

    if hasattr(provider, "_create_message"):
        try:
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_messages = [m for m in messages if m["role"] != "system"]
            response = await provider._create_message(system, user_messages)
            return response if isinstance(response, str) else None
        except Exception:
            pass

    return None


class CorrectionChain:
    """Three-stage error correction chain.

    Tries strategies in order:
    1. Deterministic cleanup — strip imports and apply AST rewrites
    2. Few-shot correction — re-invoke ``provider.generate()`` with enriched errors
    3. Fresh fixer — invoke the provider's low-level API with a specialised fixer prompt

    Attributes:
        attempts: Log of all correction attempts made during ``correct()``.
    """

    def __init__(self) -> None:
        self.attempts: list[CorrectionAttempt] = []

    async def correct(
        self,
        program: str,
        errors: list[str],
        namespace_desc: str,
        intent: str,
        allowed_names: set[str],
        provider=None,
        extra_rules: list | None = None,
    ) -> CorrectionResult | None:
        """Attempt to correct an invalid program using three strategies in order.

        Args:
            program: The invalid program that needs correction.
            errors: Validation errors from the initial program.
            namespace_desc: Formatted tool namespace string.
            intent: Original user intent.
            allowed_names: Set of allowed callable names for validation.
            provider: Inference provider instance (required for few-shot and fixer).
            extra_rules: Additional validation rules.

        Returns:
            A CorrectionResult if any strategy succeeds, or None if all fail.
        """
        # Strategy 1: Deterministic cleanup
        cleaned = deterministic_cleanup(program)
        validation = validate(cleaned, allowed_names=allowed_names, extra_rules=extra_rules)
        attempt = CorrectionAttempt(
            strategy="deterministic_cleanup",
            program=cleaned,
            errors=validation.errors,
            accepted=validation.valid,
        )
        self.attempts.append(attempt)
        if validation.valid:
            return CorrectionResult(
                program=cleaned,
                strategy="deterministic_cleanup",
                attempts=len(self.attempts),
            )

        # Strategy 2: Few-shot correction via provider.generate() with error feedback
        if provider is not None:
            enriched = enrich_errors(validation.errors, namespace_desc)
            raw = await provider.generate(
                intent, namespace_desc, error_feedback=enriched
            )
            if raw is not None:
                few_shot_program = sanitize_output(raw)
                few_shot_program = deterministic_cleanup(few_shot_program)
                fs_validation = validate(
                    few_shot_program, allowed_names=allowed_names, extra_rules=extra_rules
                )
                attempt = CorrectionAttempt(
                    strategy="few_shot_correction",
                    program=few_shot_program,
                    errors=fs_validation.errors,
                    accepted=fs_validation.valid,
                )
                self.attempts.append(attempt)
                if fs_validation.valid:
                    return CorrectionResult(
                        program=few_shot_program,
                        strategy="few_shot_correction",
                        attempts=len(self.attempts),
                    )

            # Strategy 3: Fresh fixer via low-level provider API
            errors_text = "\n".join(validation.errors)
            messages = build_fixer_messages(
                intent=intent,
                broken_program=cleaned,
                errors=errors_text,
                namespace_desc=namespace_desc,
            )
            fixer_raw = await _call_fixer(provider, messages)
            if fixer_raw is not None:
                fixer_program = sanitize_output(fixer_raw)
                fixer_program = deterministic_cleanup(fixer_program)
                fixer_validation = validate(
                    fixer_program, allowed_names=allowed_names, extra_rules=extra_rules
                )
                attempt = CorrectionAttempt(
                    strategy="fresh_fixer",
                    program=fixer_program,
                    errors=fixer_validation.errors,
                    accepted=fixer_validation.valid,
                )
                self.attempts.append(attempt)
                if fixer_validation.valid:
                    return CorrectionResult(
                        program=fixer_program,
                        strategy="fresh_fixer",
                        attempts=len(self.attempts),
                    )

        return None
