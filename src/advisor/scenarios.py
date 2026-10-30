"""Scenario generator — produce enough varied decision situations to train on.

A dataset of ``candidates x 2 mixes`` (~12 rows) is far too small to fine-tune
Laya, which needs on the order of hundreds of labeled decisions. This module
enumerates many distinct decision situations by varying the *independent
factors* that change the right index decision, so each (scenario, candidate)
pair becomes a distinct labeled row.

Varied factors (recorded per scenario for reproducibility + as context):
    * workload mix .......... baseline, shifted, plus generated read/write
      balances (``mix_variant`` perturbs per-template weights deterministically).
    * starting index state .. the candidate index may start ABSENT or already
      PRESENT. This is the only way to generate KEEP/DROP labels (not just
      CREATE/NO-OP), so both starts are enumerated.
    * data scale / distribution .. different seeds (and the Week-1 ``scale``
      factor) so selectivity / cardinality differ across scenarios.
    * per-template frequency perturbation .. a deterministic jitter applied to
      the mix weights, keyed by the scenario seed.

Every factor lives in the :class:`Scenario` descriptor so a row is fully
reproducible and the factors are available to the models as context. Generation
is deterministic for a fixed master seed, so the scenario set regenerates
identically.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from . import CONFIG_DIR

SCENARIOS_YAML = CONFIG_DIR / "scenarios.yaml"

# Starting index-state tokens.
START_ABSENT = "absent"
START_PRESENT = "present"


@dataclass(frozen=True)
class Scenario:
    """A fully-described, reproducible decision situation.

    ``scenario_id`` is stable for a given factor combination so regeneration is
    idempotent. ``weight_overrides`` holds the concrete per-template weight
    jitter so the exact mix used is reproducible without re-deriving it.
    """

    scenario_id: str
    mix: str                       # base mix name (baseline / shifted)
    mix_variant: int               # 0 == unperturbed; >0 == jittered variant
    start_state: str               # START_ABSENT or START_PRESENT
    scale: float                   # miniaturized DB scale factor
    seed: int                      # data seed for this scenario
    weight_overrides: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioConfig:
    """Generator configuration (config/scenarios.yaml)."""

    master_seed: int
    base_mixes: list[str]
    mix_variants: int              # number of jittered variants per base mix
    start_states: list[str]
    scales: list[float]
    seeds_per_combo: int           # distinct data seeds per factor combination
    jitter: float                  # max fractional perturbation of a weight

    @classmethod
    def load(cls, path: Path = SCENARIOS_YAML) -> ScenarioConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            master_seed=int(raw["master_seed"]),
            base_mixes=list(raw["base_mixes"]),
            mix_variants=int(raw["mix_variants"]),
            start_states=list(raw["start_states"]),
            scales=[float(s) for s in raw["scales"]],
            seeds_per_combo=int(raw["seeds_per_combo"]),
            jitter=float(raw["jitter"]),
        )


def _jittered_weights(
    base_weights: dict[str, float], jitter: float, rng: random.Random
) -> dict[str, float]:
    """Perturb each weight by up to +/- ``jitter`` fraction (deterministic)."""
    out: dict[str, float] = {}
    for name, w in base_weights.items():
        factor = 1.0 + rng.uniform(-jitter, jitter)
        out[name] = round(max(0.0, w * factor), 4)
    return out


def generate(
    cfg: ScenarioConfig,
    base_mix_weights: dict[str, dict[str, float]],
) -> list[Scenario]:
    """Enumerate all scenarios for ``cfg``.

    ``base_mix_weights`` maps each base mix name to its raw template weights
    (from :meth:`Workload`); only needed to compute jittered variants. The
    returned list is deterministic and sorted by ``scenario_id``.
    """
    scenarios: list[Scenario] = []
    for mix in cfg.base_mixes:
        for variant in range(cfg.mix_variants + 1):   # variant 0 == unperturbed
            for start_state in cfg.start_states:
                for scale in cfg.scales:
                    for s in range(cfg.seeds_per_combo):
                        # Derive a stable per-scenario seed from the factors.
                        combo_key = f"{mix}:{variant}:{start_state}:{scale}:{s}"
                        rng = random.Random(f"{cfg.master_seed}:{combo_key}")
                        seed = rng.randint(1, 2**31 - 1)

                        if variant == 0:
                            overrides: dict[str, float] = {}
                        else:
                            overrides = _jittered_weights(
                                base_mix_weights[mix], cfg.jitter, rng
                            )

                        scenario_id = (
                            f"{mix}-v{variant}-{start_state}"
                            f"-s{_scale_tag(scale)}-{s}"
                        )
                        scenarios.append(
                            Scenario(
                                scenario_id=scenario_id,
                                mix=mix,
                                mix_variant=variant,
                                start_state=start_state,
                                scale=scale,
                                seed=seed,
                                weight_overrides=overrides,
                            )
                        )
    scenarios.sort(key=lambda sc: sc.scenario_id)
    return scenarios


def _scale_tag(scale: float) -> str:
    """Filesystem/id-friendly tag for a scale factor (e.g. 0.001 -> 0p001)."""
    return str(scale).replace(".", "p")


def breakdown(scenarios: list[Scenario]) -> dict[str, dict[str, int]]:
    """Counts by each varied factor, for the Task-1 demo."""
    out: dict[str, dict[str, int]] = {
        "mix": {},
        "mix_variant": {},
        "start_state": {},
        "scale": {},
    }
    for sc in scenarios:
        out["mix"][sc.mix] = out["mix"].get(sc.mix, 0) + 1
        out["mix_variant"][str(sc.mix_variant)] = (
            out["mix_variant"].get(str(sc.mix_variant), 0) + 1
        )
        out["start_state"][sc.start_state] = out["start_state"].get(sc.start_state, 0) + 1
        out["scale"][str(sc.scale)] = out["scale"].get(str(sc.scale), 0) + 1
    return out


def projected_dataset_size(scenarios: list[Scenario], n_candidates: int) -> int:
    """Rows the labeled dataset will have: scenarios x candidates."""
    return len(scenarios) * n_candidates
