"""The evaluation cost score (lower is better).

    score = w_r * norm(read_runtime)
          + w_w * norm(write_overhead)
          + w_s * norm(index_storage_bytes)
          + w_b * norm(index_build_time)

    norm(x) = clamp((x - min) / (max - min), 0, 1)

Weights are tunable (config/cost_score.yaml) and sum to 1.0 by default. Every
component is ALSO reported raw so a trade-off (faster reads, slower writes) is
never hidden behind the scalar. The Week 3 labeler uses this declared cost
objective to pick each candidate's reference action (the lowest-scoring action);
this is a reference under a stated objective, not objective ground truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from . import CONFIG_DIR

COST_SCORE_YAML = CONFIG_DIR / "cost_score.yaml"

COMPONENTS = ("read_runtime", "write_overhead", "index_storage_bytes", "index_build_time")
_WEIGHT_KEYS = {
    "read_runtime": "w_r",
    "write_overhead": "w_w",
    "index_storage_bytes": "w_s",
    "index_build_time": "w_b",
}


@dataclass(frozen=True)
class CostComponents:
    """Raw (un-normalized) measured components. Units documented in the YAML."""

    read_runtime: float
    write_overhead: float
    index_storage_bytes: float
    index_build_time: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CostResult:
    """Full, transparent result: scalar score + normalized + raw components."""

    score: float
    normalized: dict[str, float]
    raw: dict[str, float]
    weights: dict[str, float]

    def breakdown(self) -> dict[str, dict[str, float]]:
        """Per-component contribution to the score, for trade-off inspection."""
        out: dict[str, dict[str, float]] = {}
        for comp in COMPONENTS:
            w = self.weights[_WEIGHT_KEYS[comp]]
            out[comp] = {
                "raw": self.raw[comp],
                "normalized": self.normalized[comp],
                "weight": w,
                "contribution": w * self.normalized[comp],
            }
        return out


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        raise ValueError(f"normalization bounds invalid: min={lo} max={hi}")
    return _clamp01((value - lo) / (hi - lo))


class CostModel:
    """Weighted cost-score computation loaded from config."""

    def __init__(self, weights: dict[str, float], bounds: dict[str, dict[str, float]]):
        self.weights = weights
        self.bounds = bounds

    @classmethod
    def load(cls, path: Path = COST_SCORE_YAML) -> CostModel:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(weights=dict(raw["weights"]), bounds=dict(raw["bounds"]))

    def weight_sum(self) -> float:
        return float(sum(self.weights.values()))

    def score(self, components: CostComponents) -> CostResult:
        """Compute the scalar score and return the full transparent breakdown."""
        raw = components.as_dict()
        normalized: dict[str, float] = {}
        total = 0.0
        for comp in COMPONENTS:
            b = self.bounds[comp]
            n = _normalize(raw[comp], float(b["min"]), float(b["max"]))
            normalized[comp] = n
            total += self.weights[_WEIGHT_KEYS[comp]] * n
        return CostResult(
            score=total,
            normalized=normalized,
            raw=raw,
            weights=dict(self.weights),
        )
