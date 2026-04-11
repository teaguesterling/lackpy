"""Tests for models.py classes."""

from ..models import User


def test_user_create():
    u = User(id=1, name="alice")
    assert u.id == 1
    assert u.name == "alice"
