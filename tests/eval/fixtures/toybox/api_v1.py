"""Legacy API — older naming convention."""

from .db import execute_sql


def get_usr(id):
    return execute_sql("SELECT * FROM users WHERE id = ?", (id,))


def save_usr(id, name):
    return execute_sql("UPDATE users SET name = ? WHERE id = ?", (name, id))


def del_usr(id):
    return execute_sql("DELETE FROM users WHERE id = ?", (id,))


def list_usrs():
    return execute_sql("SELECT * FROM users")
