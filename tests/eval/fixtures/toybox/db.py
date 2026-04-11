"""Database access helpers."""

from .errors import DatabaseError


def get_connection():
    """Open a database connection. Caller must close it."""
    return _Conn()


def execute_sql(sql, params=None):
    conn = get_connection()
    try:
        cursor = conn.execute(sql, params or ())
        return cursor.fetchall()
    finally:
        conn.close()


def leaky_query(sql):
    """Resource leak — opens a connection, never closes it."""
    conn = get_connection()
    return conn.execute(sql).fetchall()


def transaction():
    conn = get_connection()
    return _Tx(conn)


class _Conn:
    def execute(self, sql, params=()):
        return self

    def fetchall(self):
        return []

    def close(self):
        pass


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        self.conn.close()
