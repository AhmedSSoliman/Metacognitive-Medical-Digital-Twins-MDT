"""
tests/test_hypergraph/test_hypergraph.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_hypergraph.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Formalizes manual verification of:
  1. The interim rule-based safety checker's implausible-pair detection.
  2. The per-admission relative time-bucketing fix for hyperedge mining
     (a real bug: the original pd.Grouper-based bucketing split readings
     just minutes apart into different buckets purely because they
     straddled a wall-clock boundary like 06:00).
  3. That the mining algorithm correctly recovers an injected synthetic
     correlation pattern (a septic-shock-like triad) from noise.

No torch dependency -- pandas/scipy/numpy only.
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

from core.hypergraph.verification import InterimRuleBasedChecker
from core.hypergraph.construction import _binarize_abnormalities, HyperedgeMiner, HypergraphConfig


# ---------------------------------------------------------------------------
# InterimRuleBasedChecker
# ---------------------------------------------------------------------------

def test_interim_checker_accepts_plausible_combination():
    checker = InterimRuleBasedChecker()
    text = "Patient is tachycardic and hypotensive, consistent with early shock."
    assert checker.check(text) == 1.0


def test_interim_checker_flags_implausible_combination():
    checker = InterimRuleBasedChecker()
    text = "Patient is bradycardic and tachycardic simultaneously."
    score = checker.check(text)
    assert score < 1.0


def test_interim_checker_neutral_on_stable_vitals():
    checker = InterimRuleBasedChecker()
    text = "Patient is normotensive with stable vitals."
    assert checker.check(text) == 1.0


def test_interim_checker_handles_empty_text():
    checker = InterimRuleBasedChecker()
    assert checker.check("") == 0.0


# ---------------------------------------------------------------------------
# Time bucketing: the real calendar-boundary bug and its fix
# ---------------------------------------------------------------------------

def test_readings_minutes_apart_are_grouped_together_across_wallclock_boundary():
    """Regression test for a real bug: readings 2 minutes apart, straddling
    a 6am wall-clock boundary, were previously split into DIFFERENT buckets
    by pd.Grouper(freq="6h") purely because of the calendar boundary, not
    because they were actually far apart for this admission. Fixed by
    bucketing relative to each admission's own first reading instead.
    """
    df = pd.DataFrame({
        "hadm_id": [1, 1],
        "charttime": [pd.Timestamp("2023-01-01 05:59:00"), pd.Timestamp("2023-01-01 06:01:00")],
        "variable": ["heart_rate", "map"],
        "value": [115, 58],
    })
    result = _binarize_abnormalities(df, "hadm_id", window_hours=6.0)
    assert len(result) == 1, "Readings 2 minutes apart should land in exactly one bucket, not two"
    assert result.iloc[0]["heart_rate_high"] == True
    assert result.iloc[0]["map_low"] == True


def test_readings_far_apart_land_in_different_buckets():
    """Sanity check for the other direction: readings genuinely far apart
    (well beyond the window) for the SAME admission should still end up in
    different buckets -- the fix should not make bucketing meaningless by
    grouping everything into one bucket regardless of actual time gaps.
    """
    df = pd.DataFrame({
        "hadm_id": [1, 1],
        "charttime": [pd.Timestamp("2023-01-01 00:00:00"), pd.Timestamp("2023-01-02 12:00:00")],
        "variable": ["heart_rate", "map"],
        "value": [115, 58],
    })
    result = _binarize_abnormalities(df, "hadm_id", window_hours=6.0)
    assert len(result) == 2, "Readings 36 hours apart should land in different buckets"


# ---------------------------------------------------------------------------
# Hyperedge mining: recovers an injected synthetic correlation
# ---------------------------------------------------------------------------

def test_mining_recovers_injected_septic_shock_correlation():
    """Builds a synthetic dataset where 15% of admissions have an injected,
    strongly correlated triad (tachycardia + hypotension + hyperlactatemia)
    and the rest have independent random noise, then verifies the miner
    finds that exact triad as its top (lowest p-value) result.
    """
    rng = np.random.default_rng(42)
    n_admissions = 500
    records = []
    for hadm_id in range(n_admissions):
        base_time = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=hadm_id)
        is_shock = rng.random() < 0.15
        if is_shock:
            records.append({"hadm_id": hadm_id, "charttime": base_time, "variable": "heart_rate", "value": 115})
            records.append({"hadm_id": hadm_id, "charttime": base_time, "variable": "map", "value": 58})
            records.append({"hadm_id": hadm_id, "charttime": base_time, "variable": "lactate", "value": 3.2})
        else:
            records.append({"hadm_id": hadm_id, "charttime": base_time, "variable": "heart_rate",
                             "value": rng.normal(80, 10)})
            records.append({"hadm_id": hadm_id, "charttime": base_time, "variable": "map",
                             "value": rng.normal(85, 8)})
            records.append({"hadm_id": hadm_id, "charttime": base_time, "variable": "lactate",
                             "value": rng.uniform(0.5, 1.8)})

    ts = pd.DataFrame(records)
    cfg = HypergraphConfig(min_support_count=20, significance_alpha=0.01)
    miner = HyperedgeMiner(cfg)
    hyperedges = miner.mine(ts)

    assert len(hyperedges) > 0, "Expected at least one significant hyperedge to be found"
    top_hyperedge = hyperedges[0]  # sorted by p-value, so index 0 is the strongest finding
    assert set(top_hyperedge["variables"]) == {"heart_rate_high", "map_low", "lactate_high"}, (
        f"Expected the injected triad as the top hyperedge, got {top_hyperedge['variables']}"
    )
