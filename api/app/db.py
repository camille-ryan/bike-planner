"""Postgres connection helper for the live-data endpoints.

These endpoints (anchors list, cost-gradient overlay) need to query
the pgRouting tables while the preprocess wave loop is still running.
Postgres handles concurrent SELECTs against an actively-being-written
`visited` table just fine — the wave loop's INSERT/UPSERTs hold row
locks only briefly, and our queries here don't need transactional
consistency with the loop's progress.
"""
import os

import psycopg

PG_DSN = (
    f"host={os.environ.get('PGHOST', 'postgres')} "
    f"user={os.environ.get('PGUSER', 'bike')} "
    f"password={os.environ.get('PGPASSWORD', 'bike')} "
    f"dbname={os.environ.get('PGDATABASE', 'bike')}"
)


def connect() -> psycopg.Connection:
    """Open a fresh autocommit connection. The caller is responsible
    for `with conn:` or `conn.close()`. We use autocommit so each
    SELECT runs in its own implicit transaction — appropriate for
    the read-only endpoints; nothing is being written from the API."""
    return psycopg.connect(PG_DSN, autocommit=True)
