"""Tests for the ollama streaming runner."""

from unittest.mock import MagicMock

from scripts.prompt_eval.runner import GenerationRecord, generate_once


def _fake_stream(tokens: list[str]):
    """Return a generator yielding fake ollama chunks."""
    class _Msg:
        def __init__(self, c):
            self.content = c

    class _Chunk:
        def __init__(self, c, ec, pec):
            self.message = _Msg(c)
            self.eval_count = ec
            self.prompt_eval_count = pec

    for i, t in enumerate(tokens):
        yield _Chunk(t, ec=i + 1, pec=10)


def test_generate_once_returns_record():
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_stream(["hello", " world"])
    rec = generate_once(
        client=fake_client,
        model="test-model",
        system_prompt="sys",
        user_message="user",
        temperature=0.1,
        timeout=10,
    )
    assert isinstance(rec, GenerationRecord)
    assert rec.raw == "hello world"
    assert rec.model == "test-model"
    assert rec.tokens_eval == 2
    assert rec.tokens_prompt == 10
    assert rec.error is None


def test_generate_once_captures_exception():
    fake_client = MagicMock()
    fake_client.chat.side_effect = RuntimeError("boom")
    rec = generate_once(
        client=fake_client,
        model="m",
        system_prompt="s",
        user_message="u",
        temperature=0.1,
        timeout=10,
    )
    assert rec.raw == ""
    assert rec.error is not None
    assert "boom" in rec.error


def test_generate_once_reports_timeout():
    def _slow_stream():
        import time

        class _Msg:
            def __init__(self, c):
                self.content = c

        class _Chunk:
            def __init__(self, c, ec, pec):
                self.message = _Msg(c)
                self.eval_count = ec
                self.prompt_eval_count = pec

        for i in range(100):
            time.sleep(0.1)
            yield _Chunk(".", ec=i + 1, pec=5)

    fake_client = MagicMock()
    fake_client.chat.return_value = _slow_stream()
    rec = generate_once(
        client=fake_client,
        model="m",
        system_prompt="s",
        user_message="u",
        temperature=0.1,
        timeout=1,
    )
    assert rec.error is not None
    assert "timeout" in rec.error.lower() or "exceeded" in rec.error.lower()


def test_generate_once_sends_system_and_user_messages():
    """The chat call should pass the system prompt and user message correctly."""
    fake_client = MagicMock()
    fake_client.chat.return_value = _fake_stream(["ok"])
    generate_once(
        client=fake_client,
        model="m",
        system_prompt="SYS",
        user_message="USR",
        temperature=0.3,
        timeout=10,
    )
    call_kwargs = fake_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "m"
    assert call_kwargs["stream"] is True
    assert call_kwargs["options"]["temperature"] == 0.3
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "SYS"}
    assert messages[1] == {"role": "user", "content": "USR"}


def test_generate_once_handles_empty_stream():
    """A stream that yields nothing should produce an empty raw string, no error."""
    fake_client = MagicMock()
    fake_client.chat.return_value = iter([])
    rec = generate_once(
        client=fake_client,
        model="m",
        system_prompt="s",
        user_message="u",
        temperature=0.1,
        timeout=10,
    )
    assert rec.raw == ""
    assert rec.error is None
