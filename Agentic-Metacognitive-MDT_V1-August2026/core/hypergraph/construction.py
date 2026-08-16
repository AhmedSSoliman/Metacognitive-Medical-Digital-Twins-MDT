"""
core/hypergraph/construction.py

Ported unchanged from the source repo's hypergraph/construction.py
(../Agentic-DT_V1-July/hypergraph/construction.py). It has no intra-project
imports, so no import rewriting was needed; logic, thresholds, and comments
are byte-identical. References below to "data.mimic_loader" now correspond to
core/cohort/mimic.py in this layout.

Phase 3: Topological Clinical Hypergraph construction from MIMIC-IV.

A hyperedge here is a set of {variable: abnormal-direction} pairs that
co-occur significantly more often than chance within a clinically short
time window (default 6h), e.g.:
    {tachycardia, hypotension, hyperlactatemia} -> "early septic shock" hyperedge

This is co-occurrence mining, not causal discovery -- the hypergraph encodes
"these physiological derangements are known to travel together," which is
what the R_bound reward checks a generated <patient_state> against.

CRITICAL: this module must only ever be run on the DERIVATION partition
produced by data.mimic_loader.temporal_split. Running it on the evaluation
partition would leak information into the safety constraint.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)


@dataclass
class HypergraphConfig:
    co_occurrence_window_hours: float = 6.0
    min_hyperedge_order: int = 2       # minimum number of variables per hyperedge
    max_hyperedge_order: int = 5       # cap combinatorial search
    min_support_count: int = 30        # minimum number of admissions exhibiting the pattern
    significance_alpha: float = 0.01   # Fisher's exact test threshold
    output_path: str = "./hypergraph/derived_hypergraph.json"


# Abnormality direction thresholds -- these should be reviewed by IC3 clinical
# collaborators before treating the resulting hypergraph as anything more than
# a candidate for face-validity review (see build_and_review below).
ABNORMALITY_RULES = {
    "heart_rate": {"high": lambda v: v > 100, "low": lambda v: v < 60},
    "map": {"low": lambda v: v < 65},
    "sbp": {"low": lambda v: v < 90},
    "resp_rate": {"high": lambda v: v > 22, "low": lambda v: v < 10},
    "spo2": {"low": lambda v: v < 92},
    "temperature_c": {"high": lambda v: v > 38.3, "low": lambda v: v < 36.0},
    "lactate": {"high": lambda v: v > 2.0},
    "creatinine": {"high": lambda v: v > 1.5},
    "wbc": {"high": lambda v: v > 12.0, "low": lambda v: v < 4.0},
    "platelet": {"low": lambda v: v < 100.0},
    "bilirubin_total": {"high": lambda v: v > 2.0},
    "ph_arterial": {"low": lambda v: v < 7.35},
}


def _binarize_abnormalities(timeseries: pd.DataFrame, id_col: str,
                             window_hours: float = 6.0) -> pd.DataFrame:
    """Converts a long-format [id_col, charttime, variable, value] frame into
    one row per (id, time-bucket) with boolean columns for each
    'variable_direction' abnormality flag.

    IMPORTANT: buckets are computed RELATIVE TO EACH ADMISSION's own first
    reading, not aligned to wall-clock calendar boundaries. An earlier
    version used `pd.Grouper(key="charttime", freq="6h")`, which buckets by
    fixed calendar time (e.g. midnight-6am, 6am-noon, ...) -- this is a real
    bug, not a stylistic choice: two readings just 2 minutes apart can be
    split into DIFFERENT buckets purely because they straddle a wall-clock
    boundary like 06:00, systematically undercounting genuine physiological
    co-occurrences that happen to straddle a boundary. Bucketing relative to
    each admission's own timeline start avoids this specific artifact.
    """
    records = []
    for entity_id, group in timeseries.groupby(id_col):
        group = group.sort_values("charttime")
        t0 = group["charttime"].iloc[0]
        # bucket index relative to this admission's own first reading, not wall-clock time
        bucket_idx = ((group["charttime"] - t0) / pd.Timedelta(hours=window_hours)).astype(int)
        group = group.assign(_bucket=bucket_idx)

        for bucket_num, bucket_group in group.groupby("_bucket"):
            flags = {}
            for _, row in bucket_group.iterrows():
                var = row["variable"]
                val = row["value"]
                if var not in ABNORMALITY_RULES or pd.isna(val):
                    continue
                for direction, rule_fn in ABNORMALITY_RULES[var].items():
                    try:
                        if rule_fn(val):
                            flags[f"{var}_{direction}"] = True
                    except Exception:
                        continue
            if flags:
                bucket_start = t0 + bucket_num * pd.Timedelta(hours=window_hours)
                records.append({id_col: entity_id, "bucket": bucket_start, **flags})

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).fillna(False)
    return df


class HyperedgeMiner:
    def __init__(self, cfg: HypergraphConfig):
        self.cfg = cfg

    def mine(self, timeseries: pd.DataFrame, id_col: str = "hadm_id") -> list[dict]:
        """Returns a list of candidate hyperedges:
            {"variables": [...], "support": int, "p_value": float, "odds_ratio": float}
        """
        binarized = _binarize_abnormalities(timeseries, id_col, window_hours=self.cfg.co_occurrence_window_hours)
        if binarized.empty:
            logger.warning("No abnormality flags found -- check ABNORMALITY_RULES against your variable names.")
            return []

        flag_cols = [c for c in binarized.columns if c not in (id_col, "bucket")]
        n_total = len(binarized)

        hyperedges = []
        for order in range(self.cfg.min_hyperedge_order, self.cfg.max_hyperedge_order + 1):
            for combo in itertools.combinations(flag_cols, order):
                co_occurring = binarized[list(combo)].all(axis=1)
                support = int(co_occurring.sum())
                if support < self.cfg.min_support_count:
                    continue

                individual_rates = [binarized[c].mean() for c in combo]
                expected_rate = np.prod(individual_rates)
                expected_count = expected_rate * n_total

                table = [[support, n_total - support],
                         [max(int(expected_count), 1), n_total - max(int(expected_count), 1)]]
                try:
                    odds_ratio, p_value = fisher_exact(table, alternative="greater")
                except Exception:
                    continue

                if p_value < self.cfg.significance_alpha:
                    hyperedges.append({
                        "variables": list(combo),
                        "support": support,
                        "p_value": float(p_value),
                        "odds_ratio": float(odds_ratio),
                        "observed_rate": support / n_total,
                        "expected_rate_under_independence": float(expected_rate),
                    })

        hyperedges.sort(key=lambda h: h["p_value"])
        logger.info("Mined %d candidate hyperedges (orders %d-%d)",
                    len(hyperedges), self.cfg.min_hyperedge_order, self.cfg.max_hyperedge_order)
        return hyperedges


@dataclass
class FaceValidityReview:
    """Tracks the clinical face-validity review process (Manuscript 2's
    methods section requires reporting this explicitly). Populate this by
    having IC3 clinical collaborators review each mined hyperedge.
    """
    reviewed: dict = field(default_factory=dict)

    def record(self, hyperedge_idx: int, approved: bool, reviewer: str, notes: str = ""):
        self.reviewed[hyperedge_idx] = {"approved": approved, "reviewer": reviewer, "notes": notes}

    def approval_rate(self) -> float:
        if not self.reviewed:
            return 0.0
        return sum(1 for r in self.reviewed.values() if r["approved"]) / len(self.reviewed)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.reviewed, f, indent=2)


def build_and_save_hypergraph(timeseries: pd.DataFrame, cfg: HypergraphConfig,
                               id_col: str = "hadm_id") -> list[dict]:
    """End-to-end: mine hyperedges, save candidates for clinical face-validity
    review. Does NOT auto-approve anything -- the returned/saved hyperedges
    are CANDIDATES pending review (see FaceValidityReview above).
    """
    miner = HyperedgeMiner(cfg)
    hyperedges = miner.mine(timeseries, id_col=id_col)

    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_path, "w") as f:
        json.dump({
            "hyperedges": hyperedges,
            "status": "PENDING_CLINICAL_REVIEW",
            "config": cfg.__dict__,
        }, f, indent=2)
    logger.info("Saved %d candidate hyperedges to %s (status: PENDING_CLINICAL_REVIEW)",
                len(hyperedges), cfg.output_path)
    return hyperedges


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Mine candidate hyperedges from a MIMIC-IV derivation partition.")
    parser.add_argument("--timeseries_path", required=True,
                        help="Path to a parquet file of [hadm_id, charttime, variable, value] "
                             "for the DERIVATION partition only")
    parser.add_argument("--output_path", default="./hypergraph/derived_hypergraph.json")
    parser.add_argument("--min_support", type=int, default=30)
    args = parser.parse_args()

    ts = pd.read_parquet(args.timeseries_path)
    cfg = HypergraphConfig(output_path=args.output_path, min_support_count=args.min_support)
    build_and_save_hypergraph(ts, cfg)
