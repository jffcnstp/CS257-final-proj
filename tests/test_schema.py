"""Task 3 — schema: disposable DB, row-count ranges, no candidate indexes."""

from __future__ import annotations

import pytest

from advisor import schema_loader


@pytest.mark.db
def test_schema_applies_and_seeds(test_db):
    schema_loader.apply_schema(test_db)
    counts = schema_loader.seed_data(test_db, scale=0.01)  # 1% miniature

    actual = schema_loader.table_counts(test_db)
    # Proportions preserved: order_items > orders > customers.
    assert actual["order_items"] > actual["orders"] > actual["customers"] > 0
    # Seeded counts match the scaled targets.
    assert actual["customers"] == counts.customers
    assert actual["orders"] == counts.orders
    assert actual["order_items"] == counts.order_items


@pytest.mark.db
def test_no_candidate_indexes_initially(test_db):
    schema_loader.apply_schema(test_db)
    schema_loader.seed_data(test_db, scale=0.001)
    assert schema_loader.candidate_indexes_present(test_db) == []


@pytest.mark.db
def test_seed_is_deterministic(test_db):
    schema_loader.apply_schema(test_db)
    schema_loader.seed_data(test_db, scale=0.001, seed=123)
    first = schema_loader.table_counts(test_db)

    # Re-apply schema (drops + recreates tables) and reseed with same seed.
    schema_loader.apply_schema(test_db)
    schema_loader.seed_data(test_db, scale=0.001, seed=123)
    second = schema_loader.table_counts(test_db)
    assert first == second
