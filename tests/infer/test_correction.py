"""Tests for CorrectionChain."""

import pytest

from lackpy.infer.correction import CorrectionChain, CorrectionResult, CorrectionAttempt


class FakeProvider:
    def __init__(self, name, responses):
        self._name = name
        self._responses = list(responses)

    @property
    def name(self):
        return self._name

    def available(self):
        return True

    async def generate(self, intent, namespace_desc, config=None, error_feedback=None):
        return self._responses.pop(0) if self._responses else None

    async def _chat(self, messages, temperature=None):
        resp = self._responses.pop(0) if self._responses else None
        return {"message": {"content": resp}} if resp else {"message": {"content": ""}}


class TestDeterministicCleanupStrategy:
    @pytest.mark.asyncio
    async def test_strips_imports_and_validates(self):
        """Deterministic cleanup removes imports, leaving valid code."""
        chain = CorrectionChain()
        result = await chain.correct(
            program="import os\nx = 1",
            errors=["forbidden import"],
            namespace_desc="",
            intent="assign x",
            allowed_names=set(),
        )
        assert result is not None
        assert result.strategy == "deterministic_cleanup"
        assert "import" not in result.program
        assert result.program == "x = 1"

    @pytest.mark.asyncio
    async def test_records_attempt(self):
        chain = CorrectionChain()
        await chain.correct(
            program="import os\nx = 1",
            errors=["forbidden import"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
        )
        assert len(chain.attempts) >= 1
        assert chain.attempts[0].strategy == "deterministic_cleanup"
        assert chain.attempts[0].accepted is True


class TestFewShotStrategy:
    @pytest.mark.asyncio
    async def test_provider_returns_valid_code_on_retry(self):
        """Few-shot strategy uses provider.generate() with error feedback."""
        # Deterministic cleanup won't fix "bad_func()" but provider returns valid code
        provider = FakeProvider("test", ["y = 2"])
        chain = CorrectionChain()
        result = await chain.correct(
            program="bad_func()",
            errors=["unknown name: bad_func"],
            namespace_desc="",
            intent="assign y",
            allowed_names=set(),
            provider=provider,
        )
        assert result is not None
        assert result.strategy == "few_shot_correction"
        assert result.program == "y = 2"

    @pytest.mark.asyncio
    async def test_few_shot_records_attempt(self):
        provider = FakeProvider("test", ["y = 2"])
        chain = CorrectionChain()
        await chain.correct(
            program="bad_func()",
            errors=["unknown name"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
            provider=provider,
        )
        strategies = [a.strategy for a in chain.attempts]
        assert "deterministic_cleanup" in strategies
        assert "few_shot_correction" in strategies


class TestFreshFixerStrategy:
    @pytest.mark.asyncio
    async def test_provider_returns_none_for_few_shot_valid_for_fixer(self):
        """Fresh fixer is tried when few-shot returns None."""
        # First response (few-shot generate) returns None, second (_chat) returns valid code
        provider = FakeProvider("test", [None, "z = 3"])
        chain = CorrectionChain()
        result = await chain.correct(
            program="bad_func()",
            errors=["unknown name"],
            namespace_desc="",
            intent="assign z",
            allowed_names=set(),
            provider=provider,
        )
        assert result is not None
        assert result.strategy == "fresh_fixer"
        assert result.program == "z = 3"

    @pytest.mark.asyncio
    async def test_fresh_fixer_records_attempt(self):
        provider = FakeProvider("test", [None, "z = 3"])
        chain = CorrectionChain()
        await chain.correct(
            program="bad_func()",
            errors=["unknown name"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
            provider=provider,
        )
        strategies = [a.strategy for a in chain.attempts]
        assert "fresh_fixer" in strategies


class TestChainExhaustion:
    @pytest.mark.asyncio
    async def test_returns_none_when_all_strategies_fail(self):
        """Returns None when deterministic cleanup, few-shot, and fixer all fail."""
        # Both generate() and _chat() return invalid code
        provider = FakeProvider("test", ["still_bad()", "also_bad()"])
        chain = CorrectionChain()
        result = await chain.correct(
            program="bad_func()",
            errors=["unknown name"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
            provider=provider,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_without_provider(self):
        """Returns None when no provider and deterministic cleanup fails."""
        chain = CorrectionChain()
        result = await chain.correct(
            program="bad_func()",
            errors=["unknown name"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
            provider=None,
        )
        assert result is None


class TestAttemptHistory:
    @pytest.mark.asyncio
    async def test_records_all_attempts(self):
        """All three strategy attempts are recorded even when failing."""
        provider = FakeProvider("test", ["still_bad()", "also_bad()"])
        chain = CorrectionChain()
        await chain.correct(
            program="bad_func()",
            errors=["unknown name"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
            provider=provider,
        )
        # At least deterministic, few-shot, fresh fixer should all be attempted
        strategies = [a.strategy for a in chain.attempts]
        assert "deterministic_cleanup" in strategies
        assert "few_shot_correction" in strategies
        assert "fresh_fixer" in strategies

    @pytest.mark.asyncio
    async def test_attempt_accepted_flag(self):
        """Accepted flag is True only for the successful strategy."""
        chain = CorrectionChain()
        await chain.correct(
            program="import os\nx = 1",
            errors=["forbidden import"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
        )
        accepted = [a for a in chain.attempts if a.accepted]
        assert len(accepted) == 1
        assert accepted[0].strategy == "deterministic_cleanup"

    @pytest.mark.asyncio
    async def test_correction_result_attempts_count(self):
        """CorrectionResult.attempts reflects total attempts made."""
        chain = CorrectionChain()
        result = await chain.correct(
            program="import os\nx = 1",
            errors=["forbidden import"],
            namespace_desc="",
            intent="test",
            allowed_names=set(),
        )
        assert result is not None
        assert result.attempts == 1
