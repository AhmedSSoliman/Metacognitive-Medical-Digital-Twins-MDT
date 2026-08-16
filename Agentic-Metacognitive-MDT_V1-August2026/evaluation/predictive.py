"""
evaluation/predictive.py -- Endpoint 2: Deterioration Detection
(AUROC / AUPRC / precision / recall / F2), plus the bootstrap AUROC CI and
the format-compliance and deployment-feasibility sub-metrics of Endpoint 6.

Ported verbatim from the source repo's evaluation/metrics.py
(../Agentic-DT_V1-July/evaluation/metrics.py): bootstrap_auc_ci,
DeteriorationDetectionMetrics, evaluate_deterioration_detection,
evaluate_format_compliance, DeploymentMetrics, evaluate_deployment_feasibility.

WHY format-compliance and deployment-feasibility live HERE rather than in
their own file: the target tree lists five evaluation modules and allocates
no slot for Endpoint 6 (Deployment Feasibility -- schema validity, tool-call
success rate, p50/p95 latency). Its three sub-metrics are small, share no
dependency with the other endpoints, and are already reported together as
one DeploymentMetrics dataclass. They were put alongside Endpoint 2 because
both are "did the system's numeric output behave correctly" measures over
the same evaluation dataframe, and splitting them into a sixth module the
target tree never named would be a bigger deviation than grouping them.
Endpoint 4 (HITL Audit Latency) has no implementation in the source repo at
all -- it is measured by human reviewers, not code.

Imports rewritten: reward_format now comes from core.rewards.format.
"""

from __future__ import annotations

import logging

import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, fbeta_score,
)

from core.rewards.format import reward_format

logger = logging.getLogger(__name__)


def bootstrap_auc_ci(y_true: np.ndarray, y_scores: np.ndarray, n_iterations: int = 500,
                      alpha: float = 0.95, random_state: int | None = 42) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for AUROC. A single AUROC
    point estimate on a modest eval sample is misleading on its own -- this
    reports how much that estimate would plausibly move under resampling.

    Uses a LOCAL `np.random.RandomState`, not the global `np.random.seed`,
    so calling this repeatedly (e.g. once per evaluation run, or from within
    a test suite) never perturbs unrelated global random state elsewhere
    (this file's own __main__ smoke test calls np.random.seed(0)).
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true -- AUROC CI undefined, returning nan.")
        return float("nan"), float("nan")

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        # A resample can land on only one class even when the full array has
        # both (especially for small n or rare-event labels) -- skip rather
        # than crash, since AUROC is undefined for that specific resample.
        if len(np.unique(y_true[idx])) > 1:
            scores.append(roc_auc_score(y_true[idx], y_scores[idx]))

    if len(scores) < 2:
        logger.warning("Too few valid bootstrap resamples (%d) to compute a CI -- returning nan.", len(scores))
        return float("nan"), float("nan")

    lower = float(np.percentile(scores, (1.0 - alpha) / 2.0 * 100))
    upper = float(np.percentile(scores, (1.0 + alpha) / 2.0 * 100))
    return lower, upper


@dataclass
class DeteriorationDetectionMetrics:
    """Full metric set for Endpoint 2, grouped the same way DeploymentMetrics
    groups Endpoint 6's sub-metrics below -- one self-contained dataclass per
    endpoint that has more than a couple of related numbers.
    """
    auroc: float
    auprc: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    precision: float
    recall: float
    f2_score: float
    threshold: float


def evaluate_deterioration_detection(risk_scores: np.ndarray, true_labels: np.ndarray,
                                      threshold: float = 0.5, n_bootstrap: int = 500,
                                      ci_alpha: float = 0.95,
                                      random_state: int | None = 42) -> DeteriorationDetectionMetrics:
    """Endpoint 2: standard discrimination metrics for the binary
    deterioration-within-6h label (see core/cohort/mimic.py's
    compute_deterioration_labels for how the label itself is defined).

    `risk_scores` should be a continuous score extracted from the model's
    output -- e.g. from the ForecastHead regression module in
    training/backbone.py, or a simpler keyword-severity heuristic over
    <patient_state> text -- this function assumes that extraction has
    already happened elsewhere and just computes the metrics from whatever
    numeric scores it's given.

    F2 (beta=2) weights recall four times as heavily as precision: missing a
    real deterioration event is worse than a false alarm, so recall is
    prioritized over precision at the classification threshold used to turn
    the continuous risk_scores into binary predictions for precision/recall/F2
    (AUROC/AUPRC remain threshold-free, as before).
    """
    if len(np.unique(true_labels)) < 2:
        # AUROC/AUPRC/CI/precision/recall/F2 are all undefined with only one
        # class present in the GROUND TRUTH (e.g. an evaluation slice that
        # happens to contain zero deterioration events) -- return NaN
        # explicitly rather than raising, so a caller aggregating results
        # across many slices doesn't crash on this edge case. `threshold` is
        # preserved even here since it's a caller-supplied input, not a
        # computed statistic -- still worth reporting what would have been used.
        logger.warning("Only one class present in true_labels -- deterioration metrics undefined, returning nan.")
        return DeteriorationDetectionMetrics(
            auroc=float("nan"), auprc=float("nan"),
            auroc_ci_lower=float("nan"), auroc_ci_upper=float("nan"),
            precision=float("nan"), recall=float("nan"), f2_score=float("nan"),
            threshold=threshold,
        )

    auroc = float(roc_auc_score(true_labels, risk_scores))
    auprc = float(average_precision_score(true_labels, risk_scores))
    ci_lower, ci_upper = bootstrap_auc_ci(true_labels, risk_scores, n_iterations=n_bootstrap,
                                           alpha=ci_alpha, random_state=random_state)

    # zero_division=0 (sklearn's own safe default) rather than NaN here --
    # unlike single-class GROUND TRUTH above (a data-integrity problem),
    # an all-one-class PREDICTION at a given threshold is just an extreme
    # (if uninformative) classification outcome, not something to treat as
    # missing data.
    y_pred = (np.asarray(risk_scores) >= threshold).astype(int)
    precision = float(precision_score(true_labels, y_pred, zero_division=0))
    recall = float(recall_score(true_labels, y_pred, zero_division=0))
    f2 = float(fbeta_score(true_labels, y_pred, beta=2, zero_division=0))

    return DeteriorationDetectionMetrics(
        auroc=auroc, auprc=auprc, auroc_ci_lower=ci_lower, auroc_ci_upper=ci_upper,
        precision=precision, recall=recall, f2_score=f2, threshold=threshold,
    )


def find_f2_optimal_threshold(risk_scores: np.ndarray, true_labels: np.ndarray,
                               grid: np.ndarray | None = None) -> tuple[float, float]:
    """Sweeps candidate classification thresholds and returns the one that
    maximizes F2 (recall-weighted, matching evaluate_deterioration_detection's
    rationale: missing a real deterioration event is worse than a false
    alarm). The fixed 0.5 threshold used elsewhere in this module is a
    reasonable default, not a validated operating point -- this function
    exists so a caller can report what the BEST achievable F2 actually is
    for a given checkpoint's risk scores, alongside the fixed-threshold
    numbers, rather than only ever evaluating at 0.5.

    Inspired by a similar sweep in ../2026-06-15_nemotron-experiments/
    (a separate, from-scratch Nemotron-backbone experiment) -- reimplemented
    here as a small, additive function rather than ported verbatim, since
    that notebook's version used a coarse fixed 0.1-step grid; this one
    defaults to a finer 100-point grid but accepts a custom one.

    Returns (best_threshold, best_f2). If true_labels has only one class,
    returns (0.5, nan) -- F2 is undefined, matching
    evaluate_deterioration_detection's NaN convention for the same case.
    """
    true_labels = np.asarray(true_labels)
    risk_scores = np.asarray(risk_scores)
    if len(np.unique(true_labels)) < 2:
        logger.warning("Only one class present in true_labels -- F2-optimal threshold undefined, returning nan.")
        return 0.5, float("nan")

    if grid is None:
        grid = np.linspace(0.01, 0.99, 100)

    best_threshold, best_f2 = 0.5, -1.0
    for threshold in grid:
        y_pred = (risk_scores >= threshold).astype(int)
        f2 = fbeta_score(true_labels, y_pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_threshold = f2, float(threshold)

    return best_threshold, float(best_f2)


def evaluate_format_compliance(generations: list[str]) -> float:
    """Part of Endpoint 6 (Deployment Feasibility): the STRICT pass/fail
    rate of perfectly well-formed four-stream output, not an averaged
    partial-credit score. This is deliberately binary (1.0 only for a
    perfect reward_format score, 0.0 otherwise) because "schema validity
    rate" for a deployment-readiness metric should mean exactly that --
    can a downstream parser reliably consume this output or not -- rather
    than a soft, partially-credited average that would obscure how often
    a real integration would actually break.
    """
    scores = [reward_format(gen) for gen in generations]
    return float(np.mean([1.0 if s == 1.0 else 0.0 for s in scores])) if scores else 0.0


@dataclass
class DeploymentMetrics:
    schema_validity_rate: float
    tool_call_success_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


def evaluate_deployment_feasibility(latencies_ms: list[float], schema_valid_flags: list[bool],
                                     tool_call_results: list[bool]) -> DeploymentMetrics:
    """Endpoint 6 continued: latency and tool-call reliability, alongside
    the schema-validity rate above. p50/p95 (not just a mean) are reported
    because latency distributions for LLM generation are typically
    right-skewed -- a mean can look fine while a meaningful fraction of
    real requests are much slower, which p95 surfaces and a mean would hide.
    """
    latencies = np.array(latencies_ms)
    return DeploymentMetrics(
        schema_validity_rate=float(np.mean(schema_valid_flags)) if schema_valid_flags else float("nan"),
        tool_call_success_rate=float(np.mean(tool_call_results)) if tool_call_results else float("nan"),
        latency_p50_ms=float(np.percentile(latencies, 50)) if len(latencies) else float("nan"),
        latency_p95_ms=float(np.percentile(latencies, 95)) if len(latencies) else float("nan"),
    )


