"""Tasks 2 & 7 — experiment-spec completeness and cross-referencing.

Asserts the spec contains the required sections (research question, success
criteria with both accuracy and DB outcomes) and the five consolidated
elements (question, schema, workload+frequencies, candidate rules, cost
formula), and that the README links to the spec.
"""

from __future__ import annotations

import pytest

from advisor import DOCS_DIR, REPO_ROOT

SPEC = DOCS_DIR / "experiment-spec.md"
README = REPO_ROOT / "README.md"


@pytest.fixture(scope="module")
def spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


def test_spec_exists(spec_text):
    assert spec_text.strip()


@pytest.mark.parametrize(
    "heading",
    [
        "## Research Question",
        "## Success Criteria",
        "### Action-Prediction Accuracy",
        "### Real Database Outcomes",
        '### What Counts as a "Good" Result',
        "## Schema",
        "## Workload Mix and Frequencies",
        "## Candidate-Index Rules",
        "## Evaluation Cost Score",
    ],
)
def test_required_headings_present(spec_text, heading):
    assert heading in spec_text, f"missing section: {heading}"


@pytest.mark.parametrize(
    "token",
    ["macro-F1", "precision/recall", "calibration", "read runtime", "write overhead"],
)
def test_success_criteria_mentions_both_dimensions(spec_text, token):
    assert token.lower() in spec_text.lower()


@pytest.mark.parametrize(
    "element",
    [
        "CREATE",
        "KEEP",
        "DROP",
        "NO-OP",
        "customers",
        "orders",
        "order_items",
        "baseline",
        "shifted",
        "w_r",
    ],
)
def test_five_elements_consolidated(spec_text, element):
    assert element in spec_text


def test_cross_references_config(spec_text):
    for ref in ("config/schema.sql", "config/workload.yaml", "config/candidates.yaml",
                "config/cost_score.yaml"):
        assert ref in spec_text, f"spec does not cross-reference {ref}"


def test_readme_links_to_spec():
    assert "docs/experiment-spec.md" in README.read_text(encoding="utf-8")
