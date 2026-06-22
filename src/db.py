"""Postgres connection helpers."""
import psycopg
from pgvector.psycopg import register_vector
from contextlib import contextmanager
from src.config import DATABASE_URL


@contextmanager
def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
