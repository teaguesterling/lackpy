"""Tests for the Log and message types."""

from lackpy.lackey.log import Log, System, User, Assistant


class TestMessageTypes:
    def test_system_message(self):
        msg = System("You are a cell generator.")
        assert msg.role == "system"
        assert msg.content == "You are a cell generator."

    def test_user_message(self):
        msg = User("count lines in files")
        assert msg.role == "user"

    def test_assistant_accepted(self):
        msg = Assistant("files = glob('*.py')", accepted=True)
        assert msg.role == "assistant"
        assert msg.accepted is True
        assert msg.errors is None

    def test_assistant_rejected(self):
        msg = Assistant("import os", accepted=False, errors=["Forbidden AST node: Import"])
        assert msg.accepted is False
        assert len(msg.errors) == 1

    def test_assistant_with_strategy(self):
        msg = Assistant("x = 1", accepted=True, strategy="few_shot_correction")
        assert msg.strategy == "few_shot_correction"


class TestLog:
    def test_create_log(self):
        log = Log([System("prompt"), User("intent"), Assistant("code", accepted=True)])
        assert len(log.messages) == 3

    def test_log_accepted_messages(self):
        log = Log([
            System("prompt"), User("intent"),
            Assistant("bad", accepted=False, errors=["err"]),
            User("fix it"),
            Assistant("good", accepted=True),
        ])
        accepted = [m for m in log.messages if hasattr(m, "accepted") and m.accepted]
        assert len(accepted) == 1
        assert accepted[0].content == "good"

    def test_empty_log(self):
        log = Log([])
        assert len(log.messages) == 0

    def test_log_to_dicts(self):
        log = Log([System("prompt"), User("intent"), Assistant("code", accepted=True)])
        dicts = log.to_dicts()
        assert len(dicts) == 3
        assert dicts[0]["role"] == "system"
        assert dicts[2]["accepted"] is True
