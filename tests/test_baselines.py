"""Tasks 3 & 4 — baselines: documented rules on constructed rows, all actions
reachable, reproducibility. DB-independent."""

from __future__ import annotations

from advisor.baselines import NoIndexAdvisor, RuleBasedAdvisor, RuleThresholds
from advisor.candidates import CREATE, DROP, KEEP, NO_OP
from advisor.features import CandidateFeatures


def _feats(**overrides) -> CandidateFeatures:
    base = dict(
        table="orders",
        column="status",
        query_frequency=0.22,
        est_selectivity=0.1,
        est_cardinality=500_000.0,
        explain_startup_cost=0.0,
        explain_total_cost=100.0,
        explain_plan_rows=1000.0,
        index_exists=False,
        index_scans_observed=0,
        index_size_bytes=0,
        workload_write_rate=0.2,
    )
    base.update(overrides)
    return CandidateFeatures(**base)


# --------------------------- no-index advisor ---------------------------------
def test_no_index_absent_is_noop():
    d = NoIndexAdvisor().decide(_feats(index_exists=False))
    assert d.action == NO_OP


def test_no_index_present_is_drop():
    d = NoIndexAdvisor().decide(_feats(index_exists=True))
    assert d.action == DROP


def test_no_index_reproducible():
    adv = NoIndexAdvisor()
    rows = [_feats(), _feats(index_exists=True)]
    assert [d.action for d in adv.decide_all(rows)] == [d.action for d in adv.decide_all(rows)]


# --------------------------- rule-based advisor -------------------------------
def test_rule_create_when_frequent_selective_low_write():
    adv = RuleBasedAdvisor()
    d = adv.decide(_feats(query_frequency=0.3, est_selectivity=0.05, workload_write_rate=0.2))
    assert d.action == CREATE


def test_rule_noop_when_write_heavy():
    adv = RuleBasedAdvisor()
    d = adv.decide(_feats(query_frequency=0.3, est_selectivity=0.05, workload_write_rate=0.9))
    assert d.action == NO_OP


def test_rule_noop_when_not_selective():
    adv = RuleBasedAdvisor()
    d = adv.decide(_feats(est_selectivity=0.99))
    assert d.action == NO_OP


def test_rule_keep_when_index_used():
    adv = RuleBasedAdvisor()
    d = adv.decide(_feats(index_exists=True, index_scans_observed=50))
    assert d.action == KEEP


def test_rule_drop_when_index_unused():
    adv = RuleBasedAdvisor()
    d = adv.decide(_feats(index_exists=True, index_scans_observed=0))
    assert d.action == DROP


def test_all_four_actions_reachable():
    adv = RuleBasedAdvisor()
    create_row = _feats(query_frequency=0.3, est_selectivity=0.05, workload_write_rate=0.2)
    actions = {
        adv.decide(create_row).action,
        adv.decide(_feats(est_selectivity=0.99)).action,
        adv.decide(_feats(index_exists=True, index_scans_observed=50)).action,
        adv.decide(_feats(index_exists=True, index_scans_observed=0)).action,
    }
    assert actions == {CREATE, NO_OP, KEEP, DROP}


def test_thresholds_load_from_config():
    t = RuleThresholds.load()
    assert 0.0 <= t.min_query_frequency <= 1.0
    assert 0.0 <= t.max_est_selectivity <= 1.0
    assert 0.0 <= t.max_write_rate <= 1.0
    assert t.min_index_scans >= 0
