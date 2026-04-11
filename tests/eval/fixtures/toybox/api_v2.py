"""Modern API — idiomatic names."""

from .db import execute_sql


def get_user(user_id):
    return execute_sql("SELECT * FROM users WHERE id = ?", (user_id,))


def save_user(user_id, name):
    return execute_sql("UPDATE users SET name = ? WHERE id = ?", (name, user_id))


def delete_user(user_id):
    return execute_sql("DELETE FROM users WHERE id = ?", (user_id,))


def list_users():
    return execute_sql("SELECT * FROM users")
