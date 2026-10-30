"""Weight sensitivity analysis of the reference policy.

The labeler records each action's RAW cost components, so reference labels can
be re-derived under alternative cost-objective weightings **without re-running
the expensive replays**. This converts the cost objective's subjectivity from a
hidden assumption into a tested, reported property: we show how many candidates
change their reference label when the weights change.

(Model-ranking stability under each weighting is completed in Weeks 4-5 once
models exist; this module establishes the label-level sensitivity and the
re-scoring machinery.)
"""

from __future__ import annotations

from dataclasses import dataclass

from .cost import CostComponents, CostModel
from .labeler import CandidateLabel

# Named alternative weightings (w_r, w_w, w_s, w_b). The first is the default.
NAMED_WEIGHTINGS: dict[str, dict[str, float]] = {
    "read_heavy":      {"w_r": 0.5, "w_w": 0.3, "w_s": 0.1, "w_b": 0.1},
    "balanced":        {"w_r": 0.25, "w_w": 0.25, "w_s": 0.25, "w_b": 0.25},
    "write_sensitive": {"w_r": 0.3, "w_w": 0.6, "w_s": 0.05, "w_b": 0.05},
}

DEFAULT_WEIGHTING = "read_heavy"


@dataclass(frozen=True)
class WeightingReport:
    """Reference-action distribution + label-change count under one weighting."""

    weighting: str
    weights: dict[str, float]
    action_distribution: dict[str, int]
    labels_changed_vs_default: int


def _rescore_action(
    bounds: dict[str, dict[str, float]],
    weights: dict[str, float],
    components: CostComponents,
) -> float:
    """Score a single action's stored raw components under ``weights``.

    Purely arithmetic on stored components — NO replay, NO database.
    """
    model = CostModel(weights=weights, bounds=bounds)
    return model.score(components).score


def reference_action_under(
    label: CandidateLabel,
    weights: dict[str, float],
    bounds: dict[str, dict[str, float]],
) -> str:
    """Re-derive the min-cost reference action for ``label`` under ``weights``."""
    best_action = None
    best_score = float("inf")
    for action in label.allowed_actions:
        score = _rescore_action(bounds, weights, label.outcomes[action].components)
        if score < best_score:
            best_score = score
            best_action = action
    return best_action  # type: ignore[return-value]


def analyze(
    labels: list[CandidateLabel],
    *,
    weightings: dict[str, dict[str, float]] | None = None,
    default_weighting: str = DEFAULT_WEIGHTING,
    bounds: dict[str, dict[str, float]] | None = None,
) -> list[WeightingReport]:
    """Re-score all labels under each weighting and report distribution + drift.

    Reads only the stored per-action raw components (no replay). Label-change
    counts are computed relative to the default weighting's re-derived labels.
    """
    weightings = weightings or NAMED_WEIGHTINGS
    bounds = bounds or CostModel.load().bounds

    # Reference action per label under the default weighting (the comparison base).
    default_actions = [
        reference_action_under(lbl, weightings[default_weighting], bounds)
        for lbl in labels
    ]

    reports: list[WeightingReport] = []
    for name, weights in weightings.items():
        dist: dict[str, int] = {}
        changed = 0
        for lbl, default_action in zip(labels, default_actions, strict=False):
            action = reference_action_under(lbl, weights, bounds)
            dist[action] = dist.get(action, 0) + 1
            if action != default_action:
                changed += 1
        reports.append(
            WeightingReport(
                weighting=name,
                weights=dict(weights),
                action_distribution=dist,
                labels_changed_vs_default=changed,
            )
        )
    return reports
