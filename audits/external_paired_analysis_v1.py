"""Prospective six-plant analysis; pure saved-scalar helper, no study IO."""

from __future__ import annotations

import hashlib
import math
from numbers import Real
from copy import deepcopy

import numpy as np

from audits.external_transfer_adapter_v1 import COHORT_IDS, MODELS, SEEDS, require

METRICS = ("object_f1", "seg", "object_precision", "object_recall")
PLAN = dict(
    primary="object_f1 at IoU >= 0.5, complete annotated native field",
    contrast="dtpa3d_pyramid minus pyramid3d_raw",
    cohort_order=list(COHORT_IDS),
    model_order=list(MODELS),
    seed_order=list(SEEDS),
    planned_cases=36,
    plant_clusters=6,
    aggregation="arithmetic mean of three seeds within each plant, equal-plant mean",
    uncertainty=dict(
        method="paired plant percentile bootstrap; exploratory small-cluster uncertainty",
        resamples=10000,
        rng="numpy.random.Generator(numpy.random.PCG64(20261020))",
        rng_seed=20261020,
        stream="one integers(0,6,size=(10000,6),dtype=int64) draw; no prior draws",
        quantiles=[0.025, 0.975],
        quantile_method="linear",
        primary_only=True,
        paired=True,
        seed_resampling=False,
        fallback="none; flag degenerate intervals without treating them as certainty",
    ),
    missing_policy="any failed/missing/undefined case stops aggregate and CI; retain denominator36",
    sensitivity="all six leave-one-plant-out primary means, no plant removed from main result",
    secondary="SEG, object precision/recall descriptive only; other endpoints require separate prospective specification",
    p_values=False,
    population_superiority_claim=False,
    limitations=[
        "Six source plants are not a random population sample; nominal interval coverage is not established.",
        "Seeds, cells, voxels and timepoints do not increase biological n.",
        "plant1_0hrs was used for qualification; selected labels unread does not mean every plant is author-unseen.",
        "No normality-test switching, alternative interval selection or tuning after outcomes.",
    ],
)


def finite_score(value):
    require(
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
        and 0 <= value <= 1,
        "Finite defined score in [0,1] required; never silently omit/impute",
    )
    return float(value)


def summarize(rows: list[dict]) -> dict:
    """Rows must already be provenance-checked; this function certifies no file IO.

    Each row: volume_id, model, seed, status='completed', and four scalar metrics.
    Caller must verify execution/prediction receipts before supplying real scores.
    """
    expected = {(v, m, s) for v in COHORT_IDS for m in MODELS for s in SEEDS}
    observed = {}
    for row in rows:
        key = (row.get("volume_id"), row.get("model"), row.get("seed"))
        require(type(row.get("seed")) is int, "Integer computational seed required")
        require(
            key in expected and key not in observed, "Unknown/duplicate paired case"
        )
        require(
            row.get("status") == "completed",
            "Failed case: retain denominator36; no aggregate",
        )
        observed[key] = [finite_score(row.get(metric)) for metric in METRICS]
    require(
        set(observed) == expected and len(rows) == 36, "All36 planned cases required"
    )
    scores = np.asarray(
        [[[observed[v, m, s] for s in SEEDS] for m in MODELS] for v in COHORT_IDS],
        dtype=np.float64,
    )
    within = scores.mean(axis=2)  # plant,model,metric -- not 18 independent plants
    differences = within[:, 0] - within[:, 1]
    primary = differences[:, 0]
    rng = np.random.Generator(np.random.PCG64(20261020))
    before = rng.bit_generator.state
    indices = rng.integers(0, 6, size=(10000, 6), dtype=np.int64)
    distribution = primary[indices].mean(axis=1)
    interval = np.quantile(distribution, [0.025, 0.975], method="linear")
    plants = []
    for i, volume in enumerate(COHORT_IDS):
        plants.append(
            dict(
                volume_id=volume,
                seed_mean_by_model={
                    m: {
                        metric: float(within[i, j, k])
                        for k, metric in enumerate(METRICS)
                    }
                    for j, m in enumerate(MODELS)
                },
                paired_difference={
                    metric: float(differences[i, k]) for k, metric in enumerate(METRICS)
                },
                primary_by_seed={
                    str(seed): float(scores[i, 0, j, 0] - scores[i, 1, j, 0])
                    for j, seed in enumerate(SEEDS)
                },
            )
        )
    return dict(
        status="complete36_saved_scalar_paired_summary_not_provenance_certification",
        analysis_plan=deepcopy(PLAN),
        planned_cases=36,
        observed_cases=36,
        independent_sampling_unit="source plant, n=6; independence is a design assumption, not tested here",
        plants=plants,
        equal_plant_mean_by_model={
            m: {
                metric: float(within[:, j, k].mean())
                for k, metric in enumerate(METRICS)
            }
            for j, m in enumerate(MODELS)
        },
        mean_paired_difference={
            metric: float(differences[:, k].mean()) for k, metric in enumerate(METRICS)
        },
        primary_uncertainty=dict(
            nominal_level=0.95,
            interval=interval.tolist(),
            interval_degenerate=bool(interval[0] == interval[1]),
            coverage_verified=False,
            random_state_before=before,
            random_state_after=rng.bit_generator.state,
            index_matrix_sha256=hashlib.sha256(
                indices.astype("<i8").tobytes()
            ).hexdigest(),
            bootstrap_distribution_sha256=hashlib.sha256(
                distribution.astype("<f8").tobytes()
            ).hexdigest(),
        ),
        leave_one_plant_out=[
            dict(
                omitted_volume_id=v,
                mean=float(np.delete(primary, i).mean()),
                remaining_plants=5,
            )
            for i, v in enumerate(COHORT_IDS)
        ],
        units="score difference on [0,1] scale; multiply by100 for percentage points",
        external_validation_or_model_superiority_certified=False,
    )
