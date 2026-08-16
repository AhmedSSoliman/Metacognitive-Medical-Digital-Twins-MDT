"""
tests/test_cohort/test_grpo_prompts.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_build_grpo_prompts.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Tests core/cohort/grpo_prompts.py -- the bridge that fills a real schema gap
between what Phase 1's vignette builder outputs and what Phase 2's GRPO
training requires (reference_patient_state, recipient_type,
must_mention_facts -- none of which exist in Phase 1's raw output). No
torch dependency -- pandas/numpy only.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.cohort.grpo_prompts import build_grpo_prompts, GRPOPromptBuilderConfig, RECIPIENT_DISTRIBUTION


@pytest.fixture
def sample_vignettes_and_cohort():
    vignettes_df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "hadm_id": [101, 102, 103, 104],
        "prediction_time": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "prompt": ["prompt A", "prompt B", "prompt C", "prompt D"],
        "deterioration_6h": [1, 0, 1, 0],
        "hypotension_event": [1, 0, 0, 0],
        "hyperlactatemia_event": [0, 0, 1, 0],
    })
    cohort_df = pd.DataFrame({
        "stay_id": [1, 2, 3, 4],
        "hadm_id": [101, 102, 103, 104],
        "admission_type": ["EMERGENCY", "ELECTIVE", "EMERGENCY", "URGENT"],
    })
    return vignettes_df, cohort_df


def test_bridge_produces_all_required_columns(sample_vignettes_and_cohort, tmp_path):
    vignettes_df, cohort_df = sample_vignettes_and_cohort
    cfg = GRPOPromptBuilderConfig(output_path=str(tmp_path / "grpo_prompts.parquet"), seed=42)
    result = build_grpo_prompts(vignettes_df, cohort_df, cfg)

    required = {"prompt", "reference_patient_state", "recipient_type", "must_mention_facts"}
    assert required.issubset(set(result.columns)), (
        f"Missing columns Phase 2 training requires: {required - set(result.columns)}"
    )


def test_reference_state_mentions_hypotension_when_flagged(sample_vignettes_and_cohort, tmp_path):
    vignettes_df, cohort_df = sample_vignettes_and_cohort
    cfg = GRPOPromptBuilderConfig(output_path=str(tmp_path / "grpo_prompts.parquet"), seed=42)
    result = build_grpo_prompts(vignettes_df, cohort_df, cfg)

    row_with_hypotension = result[result["stay_id"] == 1].iloc[0]
    assert "hypotension" in row_with_hypotension["reference_patient_state"].lower()


def test_reference_state_describes_stable_trajectory_when_no_events(sample_vignettes_and_cohort, tmp_path):
    vignettes_df, cohort_df = sample_vignettes_and_cohort
    cfg = GRPOPromptBuilderConfig(output_path=str(tmp_path / "grpo_prompts.parquet"), seed=42)
    result = build_grpo_prompts(vignettes_df, cohort_df, cfg)

    stable_row = result[result["stay_id"] == 2].iloc[0]  # no hypotension, no hyperlactatemia
    assert "stable" in stable_row["reference_patient_state"].lower()


def test_must_mention_facts_includes_admission_type(sample_vignettes_and_cohort, tmp_path):
    vignettes_df, cohort_df = sample_vignettes_and_cohort
    cfg = GRPOPromptBuilderConfig(output_path=str(tmp_path / "grpo_prompts.parquet"), seed=42)
    result = build_grpo_prompts(vignettes_df, cohort_df, cfg)

    row = result[result["stay_id"] == 1].iloc[0]
    assert any("emergency" in fact.lower() for fact in row["must_mention_facts"])


def test_missing_admission_type_produces_empty_facts_not_a_crash():
    """Verifies the isinstance(admission_type, str) check in
    _must_mention_facts correctly handles NaN (which is truthy in Python
    but not a usable string) rather than producing a malformed
    "admission type: nan" fact string.
    """
    vignettes_df = pd.DataFrame({
        "stay_id": [1], "hadm_id": [999], "prediction_time": ["2023-01-01"],
        "prompt": ["test"], "deterioration_6h": [0],
        "hypotension_event": [0], "hyperlactatemia_event": [0],
    })
    # cohort_df has NO matching hadm_id -- left join will produce NaN admission_type
    cohort_df = pd.DataFrame({"stay_id": [2], "hadm_id": [888], "admission_type": ["ELECTIVE"]})

    cfg = GRPOPromptBuilderConfig(output_path="/tmp/test_missing_admission.parquet", seed=42)
    result = build_grpo_prompts(vignettes_df, cohort_df, cfg)
    assert result.iloc[0]["must_mention_facts"] == []


def test_recipient_type_sampling_uses_only_configured_categories(sample_vignettes_and_cohort, tmp_path):
    vignettes_df, cohort_df = sample_vignettes_and_cohort
    # Build a larger synthetic set to check the sampled categories are valid
    big_vignettes = pd.concat([vignettes_df] * 50, ignore_index=True)
    big_vignettes["stay_id"] = range(len(big_vignettes))
    big_cohort = pd.concat([cohort_df] * 50, ignore_index=True)
    big_cohort["stay_id"] = range(len(big_cohort))

    cfg = GRPOPromptBuilderConfig(output_path=str(tmp_path / "grpo_prompts.parquet"), seed=42)
    result = build_grpo_prompts(big_vignettes, big_cohort, cfg)

    assert set(result["recipient_type"].unique()).issubset(set(RECIPIENT_DISTRIBUTION.keys()))
    # With 200 samples, we should see all three categories represented at
    # least once given the configured weights (70/15/15) -- not a strict
    # statistical test, just a sanity check that sampling isn't collapsed
    # onto a single category by some bug.
    assert len(result["recipient_type"].unique()) == 3
