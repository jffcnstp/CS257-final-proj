"""Connection helpers and isolated-DB lifecycle for the index-advisor.

All experiment work runs against an *isolated, disposable* test database whose
name comes from ``INDEX_ADVISOR_DB`` (default ``index_advisor_test``). The
create/drop helpers connect to a maintenance database (``PGMAINTENANCE_DB``,
default ``postgres``) to issue ``CREATE DATABASE`` / ``DROP DATABASE`` so the
target database can be built and torn down cleanly between runs.

Nothing here touches a shared or production instance: the module only ever
creates/drops the configured disposable database.

``psycopg`` (v3) is imported lazily inside the functions that need a live
connection so that the DB-independent parts of the toolkit (and the test suite)
import cleanly on a machine without the driver or a server installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class DBConfig:
    """Connection parameters, resolved from the environment.

    ``dbname`` is the disposable test database the toolkit operates on.
    ``maintenance_db`` is the database used only to CREATE/DROP ``dbname``.
    """

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "postgres"
    dbname: str = "index_advisor_test"
    maintenance_db: str = "postgres"

    @classmethod
    def from_env(cls) -> DBConfig:
        return cls(
            host=os.environ.get("PGHOST", "localhost"),
            port=int(os.environ.get("PGPORT", "5432")),
            user=os.environ.get("PGUSER", "postgres"),
            password=os.environ.get("PGPASSWORD", "postgres"),
            dbname=os.environ.get("INDEX_ADVISOR_DB", "index_advisor_test"),
            maintenance_db=os.environ.get("PGMAINTENANCE_DB", "postgres"),
        )

    def conninfo(self, dbname: str | None = None) -> str:
        """Return a libpq connection string for ``dbname`` (defaults to target)."""
        target = dbname or self.dbname
        return (
            f"host={self.host} port={self.port} user={self.user} "
            f"password={self.password} dbname={target}"
        )


def _require_psycopg():
    try:
        import psycopg  # noqa: PLC0415  (lazy import by design)
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "psycopg is not installed. Install project deps with "
            "`pip install -e .[dev]` to use live-database features."
        ) from exc
    return psycopg


def server_available(cfg: DBConfig | None = None) -> bool:
    """Return True if the maintenance database is reachable, else False.

    Used by DB-backed tests to skip gracefully when no server is present.
    """
    cfg = cfg or DBConfig.from_env()
    try:
        psycopg = _require_psycopg()
    except RuntimeError:
        return False
    try:
        with psycopg.connect(cfg.conninfo(cfg.maintenance_db), connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@contextmanager
def connect(cfg: DBConfig | None = None, dbname: str | None = None) -> Iterator[object]:
    """Yield a psycopg connection to ``dbname`` (defaults to the target DB)."""
    cfg = cfg or DBConfig.from_env()
    psycopg = _require_psycopg()
    conn = psycopg.connect(cfg.conninfo(dbname))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_test_db(cfg: DBConfig | None = None, *, drop_existing: bool = True) -> str:
    """Create the disposable test database. Returns the database name.

    Connects to the maintenance DB in autocommit mode (CREATE/DROP DATABASE
    cannot run inside a transaction block).
    """
    cfg = cfg or DBConfig.from_env()
    psycopg = _require_psycopg()
    with psycopg.connect(cfg.conninfo(cfg.maintenance_db), autocommit=True) as conn:
        if drop_existing:
            _drop_db(conn, cfg.dbname)
        conn.execute(f'CREATE DATABASE "{cfg.dbname}"')
    return cfg.dbname


def drop_test_db(cfg: DBConfig | None = None) -> None:
    """Drop the disposable test database if it exists."""
    cfg = cfg or DBConfig.from_env()
    psycopg = _require_psycopg()
    with psycopg.connect(cfg.conninfo(cfg.maintenance_db), autocommit=True) as conn:
        _drop_db(conn, cfg.dbname)


def _drop_db(conn, dbname: str) -> None:
    """Terminate other sessions and DROP DATABASE IF EXISTS ``dbname``."""
    conn.execute(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        "WHERE datname = %s AND pid <> pg_backend_pid()",
        (dbname,),
    )
    conn.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
