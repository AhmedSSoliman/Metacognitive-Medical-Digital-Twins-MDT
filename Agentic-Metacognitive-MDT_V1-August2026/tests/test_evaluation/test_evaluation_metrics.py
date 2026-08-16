"""
tests/test_evaluation/test_evaluation_metrics.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_evaluation_metrics.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Formalizes manual verification of the additions to evaluation/report.py
(bootstrapped AUROC confidence intervals, F2-score/precision/recall for the
Deterioration Detection endpoint, and the NaN-to-null JSON fix in
save_evaluation_report) and scripts/run_evaluation.py's generate_visual_report.

The whole FILE is requires_torch, even though the logic under test in
Sections A-D (bootstrap_auc_ci, evaluate_deterioration_detection,
run_full_evaluation, save_evaluation_report) is pure numpy/pandas/
scikit-learn: evaluation/report.py itself does
`from training.backbone import parse_streams`, and training/backbone.py
imports torch/transformers/peft at module level (parse_streams itself lives
in the separate, torch-free core/parsing.py, but metrics.py imports
it via multi_stream's re-export, not directly) -- so importing
evaluation.metrics at all already requires torch, regardless of anything
this test file adds. Section E additionally needs
scripts/run_evaluation.py's generate_visual_report, which needs no model or
GPU itself, but that module also transitively imports models.multi_stream.
Both are the same underlying situation as tests/test_grpo_math.py (a file
whose *logic* is torch-free but whose *import chain* isn't), so
pytest.importorskip is placed at the very top of this file, before any
project import -- not partway through, which would raise a raw ImportError
during collection in a torch-less environment instead of skipping cleanly.
"""

import json

import numpy as np
import pandas as pd
import pytest

# PORT NOTE: the source guarded this whole file with importorskip("torch")
# because evaluation/metrics.py imported models/multi_stream.py (torch) at
# module level. After the port, evaluation/report.py and
# evaluation/predictive.py import only core.parsing / core.rewards, none of
# which import torch -- so these tests now run WITHOUT torch. The guard is
# kept (harmless where torch is installed, and Section E's docx/matplotlib
# path is still the heavier one) but the reason has changed; see below.
pytest.importorskip("torch", reason="kept from the source suite; the metric functions themselves no longer need torch after the port")

# bootstrap_auc_ci / DeteriorationDetectionMetrics / evaluate_deterioration_detection
# moved to evaluation/predictive.py in the split; run_full_evaluation /
# save_evaluation_report / EvaluationResult stayed with the aggregate in
# evaluation/report.py. Same functions, unchanged behavior.
from sklearn.metrics import fbeta_score

from evaluation.predictive import (
    bootstrap_auc_ci, DeteriorationDetectionMetrics, evaluate_deterioration_detection,
    find_f2_optimal_threshold,
)
from evaluation.report import run_full_evaluation, save_evaluation_report

pytestmark = pytest.mark.requires_torch


class _DummyChecker:
    def check(self, text):
        return 0.9


def _well_formed_generation():
    return (
        "<think>ok</think><patient_state>Tachycardic and hypotensive.</patient_state>"
        "<forecast>not applicable</forecast>"
        "<user_belief>For a clinician.</user_belief>"
    )


# ---------------------------------------------------------------------------
# Section A: bootstrap_auc_ci
# ---------------------------------------------------------------------------

def test_bootstrap_auc_ci_returns_valid_bounded_interval():
    rng = np.random.RandomState(0)
    y_true = rng.binomial(1, 0.3, 300)
    # Well-separated scores: positives skewed high, negatives skewed low.
    y_scores = np.where(y_true == 1, rng.uniform(0.5, 1.0, 300), rng.uniform(0.0, 0.5, 300))

    lower, upper = bootstrap_auc_ci(y_true, y_scores, n_iterations=200, random_state=1)
    assert 0.0 <= lower <= upper <= 1.0
    from sklearn.metrics import roc_auc_score
    point_estimate = roc_auc_score(y_true, y_scores)
    # The point estimate should fall inside (or very near) its own bootstrap CI.
    assert lower - 0.05 <= point_estimate <= upper + 0.05


def test_bootstrap_auc_ci_reproducible_with_fixed_random_state():
    rng = np.random.RandomState(0)
    y_true = rng.binomial(1, 0.4, 100)
    y_scores = rng.rand(100)

    result_a = bootstrap_auc_ci(y_true, y_scores, n_iterations=100, random_state=42)
    result_b = bootstrap_auc_ci(y_true, y_scores, n_iterations=100, random_state=42)
    assert result_a == result_b


def test_bootstrap_auc_ci_widens_with_smaller_sample():
    # Scores overlap heavily between classes (unlike the other tests' clean
    # separation) so a small slice isn't likely to land perfectly separable
    # by chance -- a perfectly separable small sample has ~zero bootstrap
    # variance (every resample scores AUROC=1.0), which would make the CI
    # narrower, not wider, defeating the point of this test.
    rng = np.random.RandomState(3)
    y_true_large = rng.binomial(1, 0.3, 500)
    y_scores_large = np.where(y_true_large == 1, rng.uniform(0.3, 0.7, 500), rng.uniform(0.3, 0.7, 500) - 0.1)
    y_true_small = y_true_large[:30]
    y_scores_small = y_scores_large[:30]

    lower_large, upper_large = bootstrap_auc_ci(y_true_large, y_scores_large, n_iterations=300, random_state=7)
    lower_small, upper_small = bootstrap_auc_ci(y_true_small, y_scores_small, n_iterations=300, random_state=7)
    assert (upper_small - lower_small) >= (upper_large - lower_large)


def test_bootstrap_auc_ci_returns_nan_for_single_class():
    y_true = np.zeros(50)
    y_scores = np.random.rand(50)
    lower, upper = bootstrap_auc_ci(y_true, y_scores)
    assert np.isnan(lower) and np.isnan(upper)


def test_bootstrap_auc_ci_does_not_perturb_global_random_state():
    # A local RandomState must be used internally -- calling this must not
    # change what np.random.* returns afterward, since other tests/smoke
    # runs in this codebase rely on np.random.seed() for their own determinism.
    np.random.seed(123)
    before = np.random.rand()
    np.random.seed(123)
    rng = np.random.RandomState(0)
    y_true = rng.binomial(1, 0.3, 100)
    y_scores = rng.rand(100)
    bootstrap_auc_ci(y_true, y_scores, n_iterations=50)
    after = np.random.rand()
    np.random.seed(123)
    expected = np.random.rand()
    assert before == expected  # sanity: seeding is deterministic
    assert after == expected   # bootstrap_auc_ci did not consume global random state


# ---------------------------------------------------------------------------
# Section B: extended evaluate_deterioration_detection
# ---------------------------------------------------------------------------

def test_evaluate_deterioration_detection_returns_dataclass_with_expected_fields():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3])
    result = evaluate_deterioration_detection(y_scores, y_true, n_bootstrap=50)
    assert isinstance(result, DeteriorationDetectionMetrics)
    field_names = {f.name for f in __import__("dataclasses").fields(result)}
    assert field_names == {
        "auroc", "auprc", "auroc_ci_lower", "auroc_ci_upper",
        "precision", "recall", "f2_score", "threshold",
    }


def test_evaluate_deterioration_detection_precision_recall_f2_hand_computed():
    # At threshold=0.5: predictions = [0, 1, 0, 1, 1, 0] (scores >= 0.5 -> 1)
    # true_labels        = [0, 1, 0, 1, 0, 1]
    # predictions        = [0, 1, 0, 1, 1, 0]
    # TP=2 (idx 1,3), FP=1 (idx 4), FN=1 (idx 5), TN=2 (idx 0,2)
    # precision = TP/(TP+FP) = 2/3, recall = TP/(TP+FN) = 2/3
    # F2 = 5 * precision * recall / (4 * precision + recall)
    #    = 5 * (2/3) * (2/3) / (4 * (2/3) + (2/3)) = (20/9) / (10/3) = 2/3
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.3])
    result = evaluate_deterioration_detection(y_scores, y_true, threshold=0.5, n_bootstrap=50)
    assert result.precision == pytest.approx(2 / 3, abs=1e-6)
    assert result.recall == pytest.approx(2 / 3, abs=1e-6)
    assert result.f2_score == pytest.approx(2 / 3, abs=1e-6)
    assert result.threshold == 0.5


def test_evaluate_deterioration_detection_threshold_changes_classification():
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.3, 0.6, 0.4, 0.55, 0.2, 0.7])

    low_threshold = evaluate_deterioration_detection(y_scores, y_true, threshold=0.2, n_bootstrap=50)
    high_threshold = evaluate_deterioration_detection(y_scores, y_true, threshold=0.8, n_bootstrap=50)
    # A very low threshold classifies almost everything positive -> high recall, lower precision.
    # A very high threshold classifies almost everything negative -> low recall.
    assert low_threshold.recall >= high_threshold.recall


def test_evaluate_deterioration_detection_single_class_all_nan_except_threshold():
    y_true = np.ones(20)
    y_scores = np.random.rand(20)
    result = evaluate_deterioration_detection(y_scores, y_true, threshold=0.7, n_bootstrap=50)
    assert np.isnan(result.auroc)
    assert np.isnan(result.auprc)
    assert np.isnan(result.auroc_ci_lower)
    assert np.isnan(result.auroc_ci_upper)
    assert np.isnan(result.precision)
    assert np.isnan(result.recall)
    assert np.isnan(result.f2_score)
    assert result.threshold == 0.7  # preserved -- it's an input, not a computed statistic


# ---------------------------------------------------------------------------
# Section B2: find_f2_optimal_threshold
# ---------------------------------------------------------------------------

def test_find_f2_optimal_threshold_beats_or_matches_fixed_default():
    y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    y_scores = np.array([0.1, 0.9, 0.3, 0.6, 0.4, 0.55, 0.85, 0.2, 0.65, 0.35])

    best_threshold, best_f2 = find_f2_optimal_threshold(y_scores, y_true)
    fixed_at_default = fbeta_score(y_true, (y_scores >= 0.5).astype(int), beta=2, zero_division=0)

    assert 0.0 < best_threshold < 1.0
    assert best_f2 >= fixed_at_default  # the sweep can never do worse than the fixed default it also tests


def test_find_f2_optimal_threshold_matches_hand_verified_grid_search():
    y_true = np.array([0, 0, 1, 1, 1])
    y_scores = np.array([0.2, 0.4, 0.5, 0.6, 0.9])
    grid = np.array([0.3, 0.55, 0.95])

    # threshold=0.3 -> preds [0,1,1,1,1] -> recall=1.0, precision=0.75 -> highest F2 on this grid
    best_threshold, best_f2 = find_f2_optimal_threshold(y_scores, y_true, grid=grid)
    assert best_threshold == 0.3
    assert best_f2 == pytest.approx(fbeta_score(y_true, [0, 1, 1, 1, 1], beta=2), abs=1e-9)


def test_find_f2_optimal_threshold_returns_nan_for_single_class():
    y_true = np.zeros(10)
    y_scores = np.random.rand(10)
    threshold, f2 = find_f2_optimal_threshold(y_scores, y_true)
    assert threshold == 0.5
    assert np.isnan(f2)


# ---------------------------------------------------------------------------
# Section C: run_full_evaluation plumbing
# ---------------------------------------------------------------------------

def _synthetic_eval_df(n=100, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "generation": [_well_formed_generation()] * n,
        "true_deterioration_label": rng.binomial(1, 0.3, n),
        "risk_score": rng.rand(n),
        "recipient_type": ["clinician"] * n,
        "latency_ms": rng.uniform(200, 900, n),
    })


def test_run_full_evaluation_deterioration_detection_field_matches_top_level():
    df = _synthetic_eval_df()
    result = run_full_evaluation(df, _DummyChecker())
    assert result.deterioration_auroc == result.deterioration_detection["auroc"]
    assert result.deterioration_auprc == result.deterioration_detection["auprc"]


def test_run_full_evaluation_respects_deterioration_threshold():
    df = _synthetic_eval_df()
    low = run_full_evaluation(df, _DummyChecker(), deterioration_threshold=0.1)
    high = run_full_evaluation(df, _DummyChecker(), deterioration_threshold=0.9)
    assert low.deterioration_detection["threshold"] == 0.1
    assert high.deterioration_detection["threshold"] == 0.9
    assert low.deterioration_detection["recall"] >= high.deterioration_detection["recall"]


# ---------------------------------------------------------------------------
# Section D: save_evaluation_report NaN/JSON fix
# ---------------------------------------------------------------------------

def test_save_evaluation_report_produces_valid_json_with_nan_fields(tmp_path):
    # A single-class eval_df makes deterioration_auroc/auprc and everything
    # inside deterioration_detection come out NaN -- exactly the case the
    # NaN-to-null fix needs to handle correctly, and would previously write
    # as the invalid `NaN` JSON token.
    df = _synthetic_eval_df()
    df["true_deterioration_label"] = 0  # force single-class
    result = run_full_evaluation(df, _DummyChecker())

    report_path = tmp_path / "report.json"
    save_evaluation_report(result, str(report_path))

    with open(report_path) as f:
        loaded = json.load(f)  # raises if the file isn't valid JSON

    assert loaded["deterioration_auroc"] is None
    assert loaded["deterioration_detection"]["auroc"] is None
    assert loaded["deterioration_detection"]["threshold"] == 0.5


# ---------------------------------------------------------------------------
# Section E: generate_visual_report (requires torch via module import, no
# model/GPU actually needed -- see module docstring; the file-level
# importorskip/pytestmark at the top of this file already cover this section)
# ---------------------------------------------------------------------------

# PORT NOTE: the source side-loaded scripts/run_evaluation.py by file path
# via importlib, because it lived under scripts/ (not an importable package)
# AND imported torch at module level. It is now evaluation/run_evaluation.py
# with its model import deferred into main(), so this is a plain import.
from evaluation.run_evaluation import generate_visual_report

from evaluation.report import EvaluationResult


def _synthetic_result():
    return EvaluationResult(
        topological_fidelity=0.8,
        deterioration_auroc=0.75,
        deterioration_auprc=0.6,
        trajectory_mae=None,
        structural_empathy_mean=0.7,
        format_compliance_rate=0.95,
        deployment={
            "schema_validity_rate": 0.95, "tool_call_success_rate": 1.0,
            "latency_p50_ms": 400.0, "latency_p95_ms": 900.0,
        },
        n_examples=50,
        deterioration_detection={
            "auroc": 0.75, "auprc": 0.6, "auroc_ci_lower": 0.65, "auroc_ci_upper": 0.85,
            "precision": 0.7, "recall": 0.65, "f2_score": 0.66, "threshold": 0.5,
        },
        f2_optimal_threshold=0.42,
        f2_at_optimal_threshold=0.71,
    )


def test_generate_visual_report_creates_nonempty_png_and_docx(tmp_path):
    result = _synthetic_result()
    eval_df = _synthetic_eval_df()
    png_path, docx_path = generate_visual_report(result, eval_df, output_dir=tmp_path, report_name="test_report")
    assert png_path.exists() and png_path.stat().st_size > 0
    assert docx_path.exists() and docx_path.stat().st_size > 0


def test_generate_visual_report_docx_contains_all_result_fields(tmp_path):
    from docx import Document
    result = _synthetic_result()
    eval_df = _synthetic_eval_df()
    _, docx_path = generate_visual_report(result, eval_df, output_dir=tmp_path, report_name="test_report")

    doc = Document(str(docx_path))
    all_cell_text = " ".join(
        cell.text for table in doc.tables for row in table.rows for cell in row.cells
    )
    for key in result.__dict__:
        assert key in all_cell_text, f"Expected '{key}' to appear in the report table"


def test_generate_visual_report_handles_none_trajectory_mae(tmp_path):
    result = _synthetic_result()
    assert result.trajectory_mae is None
    eval_df = _synthetic_eval_df()
    png_path, docx_path = generate_visual_report(result, eval_df, output_dir=tmp_path, report_name="test_report")
    assert png_path.exists() and docx_path.exists()
