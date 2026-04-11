"""Tests for auth.py helpers."""

from ..auth import hash_password


def test_hash_password():
    h = hash_password("hunter2")
    assert len(h) == 64
