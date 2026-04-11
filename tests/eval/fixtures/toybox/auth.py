"""Authentication helpers: token validation, hashing, permissions."""

import hashlib
from .db import execute_sql
from .errors import AuthError


def deprecated(fn):
    """Placeholder decorator; no-op for static analysis."""
    return fn


def validate_token(token):
    if not token:
        raise AuthError("missing token")
    row = execute_sql("SELECT user_id FROM sessions WHERE token = ?", (token,))
    if not row:
        raise AuthError("invalid token")
    return row[0]


def refresh_token(old_token):
    user_id = validate_token(old_token)
    new_token = hash_password(str(user_id) + "salt")
    execute_sql("INSERT INTO sessions (user_id, token) VALUES (?, ?)",
                (user_id, new_token))
    return new_token


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def _hash_password(password, salt):
    return hashlib.sha256((password + salt).encode()).hexdigest()


def check_permissions(token, action):
    user_id = validate_token(token)
    row = execute_sql("SELECT role FROM users WHERE id = ?", (user_id,))
    if not row or row[0] != "admin":
        raise AuthError("forbidden")
    return True


@deprecated
def parse_legacy_token(raw):
    parts = raw.split(":")
    return {"user": parts[0], "token": parts[1]}
