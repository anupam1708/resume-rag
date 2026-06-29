"""
Neo4j connection helpers (Graph RAG add-on).

A single shared driver (thread-safe, connection-pooled) plus a session
contextmanager — the Neo4j analogue of db.py's get_conn() for Postgres.

Don't open raw drivers elsewhere; go through get_session().
"""
from contextlib import contextmanager

from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

_driver = None


def get_driver():
    """Lazily create the shared driver. The driver is a long-lived, pooled object."""
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _driver


@contextmanager
def get_session():
    """Yield a Neo4j session, closed on exit. Open one per unit of work."""
    session = get_driver().session()
    try:
        yield session
    finally:
        session.close()


def close_driver():
    """Close the shared driver (call on process shutdown if you want clean exit)."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
