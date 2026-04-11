"""Tests for app.py routes."""

from ..app import login, get_user_view


def test_login_flow():
    result = login({"username": "alice", "password": "pw"})
    assert result == {"ok": True}


def test_user_list():
    result = get_user_view({"token": "t"}, 1)
    assert result is not None
