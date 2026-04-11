"""Route layer. Routes call auth and db layers directly."""

from .auth import validate_token, check_permissions
from .db import execute_sql, get_connection
from .errors import AuthError, ValidationError


def route(path):
    """Placeholder @route decorator; no-op for static analysis."""
    def wrap(fn):
        return fn
    return wrap


@route("/login")
def login(request):
    username = request.get("username")
    password = request.get("password")
    # SQL-concat smell #1
    row = execute_sql("SELECT id FROM users WHERE name = '" + username + "'")
    if not row:
        raise AuthError("unknown user")
    return {"ok": True}


@route("/users/<id>")
def get_user_view(request, id):
    token = request.get("token")
    validate_token(token)
    # SQL-concat smell #2
    return execute_sql("SELECT * FROM users WHERE id = " + str(id))


@route("/users/<id>/delete")
def delete_user_view(request, id):
    token = request.get("token")
    validate_token(token)
    check_permissions(token, "delete")
    # SQL-concat smell #3
    return execute_sql("DELETE FROM users WHERE id = " + str(id))


@route("/health")
def health(request):
    conn = get_connection()
    return {"ok": True, "db": bool(conn)}
