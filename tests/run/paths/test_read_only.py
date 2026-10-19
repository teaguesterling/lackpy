"""``read_only_policy``: the read-only literate ceiling as a policy (literate stepy D7, Ruling F).

The grade ceiling becomes a policy: a read-only profile adds a deny on every
write (``Edit(//**)``). Reads keep the policy they had.
"""

from pathlib import Path

import pytest

from lackpy.run.paths import load_policy, read_only_policy
from lackpy.run.refusal import LackpyRefusal


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    (w / "a.txt").write_text("a")
    return w


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    (h / ".claude").mkdir(parents=True)
    return h


def test_every_write_refuses(ws, home):
    ro = read_only_policy(load_policy(ws, home=home, managed=None))
    for path in (ws / "a.txt", ws / "new.txt", ws / "sub" / "deep.txt", ws):
        with pytest.raises(LackpyRefusal, match=r"Edit\(//\*\*\)"):
            ro.check(path, "write")


def test_reads_keep_the_policy(ws, home):
    ro = read_only_policy(load_policy(ws, home=home, managed=None))
    ro.check(ws / "a.txt", "read")  # inside the workspace: allowed
    with pytest.raises(LackpyRefusal):
        ro.check(Path("/etc/hostname"), "read")  # outside: still refused


def test_the_denies_of_the_policy_stay(ws, home):
    (ws / ".claude").mkdir()
    (ws / ".claude" / "settings.json").write_text(
        '{"permissions": {"deny": ["Read(secret.txt)"]}}')
    (ws / "secret.txt").write_text("s")
    ro = read_only_policy(load_policy(ws, home=home, managed=None))
    with pytest.raises(LackpyRefusal, match=r"Read\(secret.txt\)"):
        ro.check(ws / "secret.txt", "read")


def test_the_given_policy_is_not_changed(ws, home):
    policy = load_policy(ws, home=home, managed=None)
    read_only_policy(policy)
    policy.check(ws / "a.txt", "write")  # still writable


def test_the_workspace_and_extra_directories_are_kept(ws, home, tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "e.txt").write_text("e")
    (ws / ".claude").mkdir()
    (ws / ".claude" / "settings.json").write_text(
        '{"permissions": {"additionalDirectories": ["%s"]}}' % extra)
    policy = load_policy(ws, home=home, managed=None)
    ro = read_only_policy(policy)
    assert ro.workspace == policy.workspace
    ro.check(extra / "e.txt", "read")
    with pytest.raises(LackpyRefusal):
        ro.check(extra / "e.txt", "write")


def test_the_protected_list_is_not_doubled(ws, home):
    policy = load_policy(ws, home=home, managed=None)
    ro = read_only_policy(policy)
    assert ro.protected == policy.protected
    assert len(ro.protected) == len(set(ro.protected))
