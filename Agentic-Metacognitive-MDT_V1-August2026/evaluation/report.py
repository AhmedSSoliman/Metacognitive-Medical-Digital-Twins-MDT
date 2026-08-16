"""
evaluation/report.py

The consolidated EvaluationResult object, the run_full_evaluation driver that
calls every endpoint in one pass, and JSON report serialization.

PORTING NOTE (2026-08-12): ported verbatim from the source repo's
evaluation/metrics.py (../Agentic-DT_V1-July/evaluation/metrics.py):
EvaluationResult, run_full_evaluation, _nan_to_none, save_evaluation_report,
and the module __main__ smoke test.

The target tree splits metrics.py's ENDPOINT FUNCTIONS across topological.py /
predictive.py / retention.py / communication.py but allocates no slot for the
aggregate that ties them together. run_full_evaluation is the actual entry
point every SLURM evaluation job goes through, so dropping it was not an
option and duplicating it into one of the endpoint files would make that file
import all the others. It lives here, in the one place that imports from all
four endpoint modules.

ORIGINAL evaluation/metrics.py MODULE DOCSTRING:

    Implements the six evaluation endpoints from the proposal's Evaluation Plan:
      1. Topological Fidelity       -- R_bound score averaged over held-out generations
      2. Deterioration Detection    -- AUROC/AUPRC vs. the 6h deterioration label
      3. Trajectory Accuracy        -- MAE of any forecasted numeric values
      4. HITL Audit Latency         -- reviewer time vs. an unconstrained baseline
      5. Structural Empathy         -- R_emp score across recipient types
      6. Deployment Feasibility     -- schema validity, tool-call success rate, p50/p95 latency

    Run this against a held-out EVALUATION partition only (never the derivation
    partition used to mine the hypergraph) -- mixing the two would let
    information the hypergraph was built from leak into its own evaluation.

Where each endpoint now lives:
  1 -> evaluation/topological.py     4 -> (no code; human-measured)
  2 -> evaluation/predictive.py      5 -> evaluation/communication.py
  3 -> evaluation/retention.py       6 -> evaluation/predictive.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.communication import evaluate_structural_empathy
from evaluation.predictive import (
    evaluate_deployment_feasibility,
    evaluate_deterioration_detection,
    evaluate_format_compliance,
    find_f2_optimal_threshold,
)
from evaluation.retention import evaluate_trajectory_accuracy
from evaluation.topological import evaluate_topological_fidelity

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """One consolidated result object covering all six endpoints, so a full
    evaluation run produces a single, easy-to-log/serialize object rather
    than six separate return values scattered across the caller's code.
    """
    topological_fidelity: float
    deterioration_auroc: float
    deterioration_auprc: float
    trajectory_mae: float | None   # None (not NaN) when no forecast columns were present at all --
                                    # distinguishes "not computed because the data wasn't there" from
                                    # "computed but came out NaN" (e.g. all predictions were missing)
    structural_empathy_mean: float
    format_compliance_rate: float
    deployment: dict
    n_examples: int
    deterioration_detection: dict  # DeteriorationDetectionMetrics.__dict__ -- auroc/auprc are
                                    # intentionally duplicated from the two flat fields above (kept
                                    # for backward compatibility with the old flat JSON shape) plus
                                    # auroc_ci_lower, auroc_ci_upper, precision, recall, f2_score,
                                    # threshold -- everything the flat fields don't carry
    f2_optimal_threshold: float     # best threshold found by sweeping, NOT what precision/recall/f2
                                    # above were computed at (those use deterioration_threshold, a
                                    # fixed caller-supplied default of 0.5) -- report this ALONGSIDE
                                    # the fixed-threshold numbers, never as a silent replacement for
                                    # them, since picking a threshold post-hoc on the eval set itself
                                    # is optimistic and shouldn't be reported as if pre-registered.
    f2_at_optimal_threshold: float  # the F2 achieved at f2_optimal_threshold, for direct comparison
                                    # against deterioration_detection['f2_score'] (the fixed-threshold one)


def run_full_evaluation(
    eval_df: pd.DataFrame,  # expects columns: generation, true_deterioration_label,
                            # risk_score, recipient_type, latency_ms, and OPTIONALLY
                            # predicted_value/true_value (forecast accuracy) and
                            # schema_valid/tool_success (deployment feasibility) --
                            # the latter two default to all-True if absent, which
                            # means "assume clean" rather than "assume nothing
                            # deployment-related was tested"; be aware of this
                            # default if you're calling this on a dataset that
                            # never actually recorded these columns.
    hypergraph_checker,
    deterioration_threshold: float = 0.5,
    n_bootstrap: int = 500,
    ci_alpha: float = 0.95,
) -> EvaluationResult:
    """Runs all six endpoints in one pass over a single evaluation dataframe
    and returns one consolidated EvaluationResult. Splitting the individual
    evaluate_* functions out above (rather than inlining everything here)
    lets each endpoint be unit-tested or re-run independently -- e.g. if you
    only want to recheck Topological Fidelity after swapping in a newly
    reviewed hypergraph, without recomputing the other five endpoints.
    """
    generations = eval_df["generation"].tolist()

    topo_fidelity = evaluate_topological_fidelity(generations, hypergraph_checker)
    det_metrics = evaluate_deterioration_detection(
        eval_df["risk_score"].to_numpy(), eval_df["true_deterioration_label"].to_numpy(),
        threshold=deterioration_threshold, n_bootstrap=n_bootstrap, ci_alpha=ci_alpha,
    )
    f2_opt_threshold, f2_at_opt = find_f2_optimal_threshold(
        eval_df["risk_score"].to_numpy(), eval_df["true_deterioration_label"].to_numpy(),
    )

    # Trajectory accuracy is OPTIONAL, not required, because early Phase 1/2
    # runs (before a forecast-extraction mechanism -- ForecastHead or
    # text-parsed <forecast> values -- is wired up end-to-end) legitimately
    # have nothing to compute this from yet; None signals "not attempted"
    # rather than silently reporting a misleading 0.0 or crashing on a
    # missing column.
    trajectory_mae = None
    if "predicted_value" in eval_df.columns and "true_value" in eval_df.columns:
        trajectory_mae = evaluate_trajectory_accuracy(
            eval_df["predicted_value"].to_numpy(), eval_df["true_value"].to_numpy()
        )

    empathy = evaluate_structural_empathy(generations, eval_df["recipient_type"].tolist())
    format_rate = evaluate_format_compliance(generations)
    deployment = evaluate_deployment_feasibility(
        eval_df["latency_ms"].tolist(),
        eval_df.get("schema_valid", pd.Series([True] * len(eval_df))).tolist(),
        eval_df.get("tool_success", pd.Series([True] * len(eval_df))).tolist(),
    )

    result = EvaluationResult(
        topological_fidelity=topo_fidelity,
        deterioration_auroc=det_metrics.auroc,
        deterioration_auprc=det_metrics.auprc,
        trajectory_mae=trajectory_mae,
        structural_empathy_mean=empathy,
        format_compliance_rate=format_rate,
        deployment=deployment.__dict__,
        n_examples=len(eval_df),
        deterioration_detection=det_metrics.__dict__,
        f2_optimal_threshold=f2_opt_threshold,
        f2_at_optimal_threshold=f2_at_opt,
    )
    return result


def _nan_to_none(obj):
    """Recursively replaces float NaN with None so json.dump produces valid
    JSON. NOTE: a previous version of save_evaluation_report below tried to
    do this via json.dump's `default=` callback, but that never actually
    fired -- json's own encoder handles float values, including NaN,
    natively before ever falling back to `default`, so NaN was silently
    written as the non-standard `NaN` token instead of `null`. This
    preprocessing pass actually works (verified by round-tripping through
    json.loads); `save_evaluation_report` also now passes `allow_nan=False`
    so any NaN that somehow slips through this pass raises loudly instead of
    silently producing invalid JSON again.
    """
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def save_evaluation_report(result: EvaluationResult, path: str):
    with open(path, "w") as f:
        json.dump(_nan_to_none(result.__dict__), f, indent=2, allow_nan=False)
    logger.info("Saved evaluation report to %s", path)


if __name__ == "__main__":
    # Smoke test with synthetic data (no real model or MIMIC-IV data needed).
    # NOTE: the synthetic `generation` string below MUST include all four
    # stream tags (think, patient_state, forecast, user_belief) to be scored
    # as well-formed by parse_streams -- an earlier version of this smoke
    # test predated the forecast stream being added to the architecture and
    # was missing the <forecast> tag, which would have made
    # evaluate_format_compliance silently report 0% compliance even though
    # nothing was actually wrong with the reward logic itself. Caught and
    # fixed while adding these comments -- a good example of why the
    # smoke test's own fixtures need to be kept in sync with the model
    # whenever the stream format changes.
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "generation": [
            "<think>ok</think><patient_state>Tachycardic and hypotensive.</patient_state>"
            "<forecast>not applicable</forecast>"
            "<user_belief>For a clinician.</user_belief>"
        ] * n,
        "true_deterioration_label": np.random.binomial(1, 0.2, n),
        "risk_score": np.random.rand(n),
        "recipient_type": ["clinician"] * n,
        "latency_ms": np.random.uniform(200, 900, n),
    })

    class _DummyChecker:
        def check(self, text):
            return 0.9

    result = run_full_evaluation(df, _DummyChecker())
    print(result)
    assert result.format_compliance_rate == 1.0, (
        "Expected 100% format compliance with a well-formed synthetic generation "
        "including all four stream tags -- if this fails, check that the smoke "
        "test's fixture text matches training/backbone.py's current STREAM_TAGS."
    )
    print("Smoke test passed: format_compliance_rate == 1.0 as expected.")
