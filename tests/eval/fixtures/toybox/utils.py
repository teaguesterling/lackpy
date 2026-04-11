"""Utility helpers."""

import hashlib


def deprecated(fn):
    return fn


@deprecated
def parse_date(s):
    parts = s.split("-")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def format_user(user):
    return f"{user.id}: {user.name}"


def compute_hash(data):
    """Duplicate of auth.hash_password — dead code."""
    return hashlib.sha256(data.encode()).hexdigest()
