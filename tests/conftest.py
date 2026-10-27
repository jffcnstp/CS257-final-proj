"""Shared pytest fixtures.

DB-backed tests are marked ``@pytest.mark.db`` and skipped automatically when no
PostgreSQL server is reachable, so the suite stays green on a machine without a
database. CI (from Week 2) provides a Postgres service so those tests run for
real.
"""

from __future__ import annotations

import pytest

from advisor.db import DBConfig, create_test_db, drop_test_db, server_available


@pytest.fixture(scope="session")
def db_available() -> bool:
    return server_available(DBConfig.from_env())


@pytest.fixture()
def test_db(db_available):
    """Provide a freshly-created disposable test DB; drop it afterwards."""
    if not db_available:
        pytest.skip("no reachable PostgreSQL server")
    cfg = DBConfig.from_env()
    create_test_db(cfg, drop_existing=True)
    try:
        yield cfg
    finally:
        drop_test_db(cfg)
