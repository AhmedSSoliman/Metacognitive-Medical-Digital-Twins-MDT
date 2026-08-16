"""
data/preprocessing.py

Converts joined MIMIC-IV cohort + time series + notes into text-encoded
"clinical vignettes": a serialized patient-state representation used as
model input, plus the target labels needed for SFT and downstream RL reward
computation (deterioration events, mortality, etc.).

This module also builds the two-tier data split described in Manuscript 3's
plan: tier-one (fine-tuning text, e.g. Medical-O1-Reasoning-SFT, loaded
separately -- not from MIMIC-IV) is never merged with tier-two (MIMIC-IV,
used here only for hypergraph derivation and evaluation).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VignetteConfig:
    lookback_hours: int = 24          # how much history to include per vignette
    prediction_horizon_hours: int = 6  # how far ahead the deterioration label looks
    max_events_per_variable: int = 12  # cap on how many timestamped readings to serialize
    deterioration_lactate_threshold: float = 2.0     # mmol/L, hyperlactatemia
    deterioration_map_threshold: float = 65.0        # mmHg, hypotension
    output_dir: str = "./data/processed"


class VignetteBuilder:
    """Builds one text vignette per (stay_id, prediction_time) sample."""

    def __init__(self, cfg: VignetteConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # -- Label construction ---------------------------------------------------

    def compute_deterioration_labels(self, vitals: pd.DataFrame, labs: pd.DataFrame,
                                      prediction_time: pd.Timestamp,
                                      stay_id: int, hadm_id: int) -> dict:
        """Binary deterioration label over the prediction horizon: hemodynamic
        instability (MAP below threshold) OR rising lactate above threshold,
        whichever occurs first within the horizon window.

        This is intentionally a simple, physiologically interpretable rule
        for the SFT/RL reward target -- NOT the Aim 3 hypergraph itself,
        which is mined separately and at a different level of structure.

        `vitals`/`labs` are expected to already be sliced down to this
        stay_id/hadm_id (see build_vignettes_for_cohort's per-admission
        pre-partitioning) -- this only applies the time-window filter, not a
        stay_id/hadm_id filter, so callers must not pass the full cohort-wide
        tables here.
        """
        horizon_end = prediction_time + pd.Timedelta(hours=self.cfg.prediction_horizon_hours)

        future_vitals = vitals[
            (vitals["charttime"] > prediction_time)
            & (vitals["charttime"] <= horizon_end)
        ]
        future_labs = labs[
            (labs["charttime"] > prediction_time)
            & (labs["charttime"] <= horizon_end)
        ]

        hypotension = future_vitals[
            (future_vitals["variable"] == "map") & (future_vitals["value"] < self.cfg.deterioration_map_threshold)
        ]
        hyperlactatemia = future_labs[
            (future_labs["variable"] == "lactate") & (future_labs["value"] > self.cfg.deterioration_lactate_threshold)
        ]

        label = int(len(hypotension) > 0 or len(hyperlactatemia) > 0)
        return {
            "deterioration_6h": label,
            "hypotension_event": int(len(hypotension) > 0),
            "hyperlactatemia_event": int(len(hyperlactatemia) > 0),
        }

    # -- Serialization ----------------------------------------------------------

    def _serialize_timeseries(self, df: pd.DataFrame, time_col: str,
                               prediction_time: pd.Timestamp) -> str:
        """Turns a filtered time-series slice into a compact text block, e.g.:
            heart_rate: 88 (-6h), 94 (-3h), 101 (-1h)
            map: 72 (-6h), 68 (-3h), 61 (-1h)

        `df` is expected to already be sliced down to a single admission
        (see build_vignettes_for_cohort's per-admission pre-partitioning) --
        this only applies the time-window filter.
        """
        lookback_start = prediction_time - pd.Timedelta(hours=self.cfg.lookback_hours)
        window = df[(df[time_col] > lookback_start) & (df[time_col] <= prediction_time)]

        lines = []
        for var, group in window.groupby("variable"):
            group = group.sort_values(time_col).tail(self.cfg.max_events_per_variable)
            points = []
            for _, row in group.iterrows():
                hours_ago = (prediction_time - row[time_col]).total_seconds() / 3600.0
                points.append(f"{row['value']:.1f} (-{hours_ago:.1f}h)")
            lines.append(f"{var}: " + ", ".join(points))
        return "\n".join(lines) if lines else "(no data in lookback window)"

    def build_vignette(self, stay_row: pd.Series, vitals: pd.DataFrame, labs: pd.DataFrame,
                        prediction_time: pd.Timestamp) -> dict:
        """`vitals`/`labs` must already be sliced down to this stay_id/hadm_id
        (see build_vignettes_for_cohort's per-admission pre-partitioning) --
        this does not filter by stay_id/hadm_id itself.
        """
        stay_id = int(stay_row["stay_id"])
        hadm_id = int(stay_row["hadm_id"])

        vitals_text = self._serialize_timeseries(vitals, "charttime", prediction_time)
        labs_text = self._serialize_timeseries(labs, "charttime", prediction_time)

        demographics = (
            f"Age: {stay_row.get('age_at_admission', 'unknown')}, "
            f"Sex: {stay_row.get('gender', 'unknown')}, "
            f"Admission type: {stay_row.get('admission_type', 'unknown')}"
        )

        prompt = (
            "You are assisting with ICU patient monitoring. Given the patient's demographics "
            "and recent vitals/labs, reason about their current physiological state.\n\n"
            f"### Demographics\n{demographics}\n\n"
            f"### Vitals (last {self.cfg.lookback_hours}h)\n{vitals_text}\n\n"
            f"### Labs (last {self.cfg.lookback_hours}h)\n{labs_text}\n\n"
            f"### Task\nAssess this patient's risk of deterioration in the next "
            f"{self.cfg.prediction_horizon_hours} hours."
        )

        labels = self.compute_deterioration_labels(vitals, labs, prediction_time, stay_id, hadm_id)

        return {
            "stay_id": stay_id,
            "hadm_id": hadm_id,
            "prediction_time": str(prediction_time),
            "prompt": prompt,
            **labels,
        }

    def build_vignettes_for_cohort(self, cohort: pd.DataFrame, vitals: pd.DataFrame,
                                    labs: pd.DataFrame, samples_per_stay: int = 3,
                                    seed: int = 42) -> pd.DataFrame:
        """Samples `samples_per_stay` prediction times per ICU stay (uniformly
        within the stay, after the initial lookback window has elapsed) and
        builds a vignette for each.

        Pre-partitions `vitals`/`labs` by stay_id/hadm_id ONCE up front,
        rather than re-scanning the full cohort-wide tables inside the loop.
        This is not a micro-optimization: build_vignette's per-sample cost
        used to include four full linear scans over the entire multi-
        million-row vitals/labs tables (two inside _serialize_timeseries,
        two inside compute_deterioration_labels), repeated once per
        (stay, sample) pair -- with a 90k+ stay cohort at samples_per_stay=3,
        that is well over a million full-table scans, and is what caused a
        real 24h SLURM job to time out with zero output despite having
        already finished the (expensive but one-time) raw CSV scan. Grouping
        once turns each per-admission lookup into an O(1) dict access.
        """
        vitals_by_stay = {k: v for k, v in vitals.groupby("stay_id")}
        labs_by_hadm = {k: v for k, v in labs.groupby("hadm_id")}
        empty_vitals = vitals.iloc[0:0]
        empty_labs = labs.iloc[0:0]

        rng = np.random.default_rng(seed)
        records = []
        for _, stay_row in cohort.iterrows():
            intime = stay_row["intime"]
            outtime = stay_row["outtime"]
            earliest = intime + pd.Timedelta(hours=self.cfg.lookback_hours)
            latest = outtime - pd.Timedelta(hours=self.cfg.prediction_horizon_hours)
            if earliest >= latest:
                continue

            stay_id = int(stay_row["stay_id"])
            hadm_id = int(stay_row["hadm_id"])
            stay_vitals = vitals_by_stay.get(stay_id, empty_vitals)
            stay_labs = labs_by_hadm.get(hadm_id, empty_labs)

            span_hours = (latest - earliest).total_seconds() / 3600.0
            offsets = rng.uniform(0, span_hours, size=samples_per_stay)
            for offset in offsets:
                pred_time = earliest + pd.Timedelta(hours=float(offset))
                try:
                    records.append(self.build_vignette(stay_row, stay_vitals, stay_labs, pred_time))
                except Exception as e:
                    logger.warning("Skipping stay_id=%s at %s due to error: %s",
                                   stay_row.get("stay_id"), pred_time, e)

        df = pd.DataFrame(records)
        out_path = Path(self.cfg.output_dir) / "vignettes.parquet"
        df.to_parquet(out_path)
        logger.info("Built %d vignettes -> %s", len(df), out_path)
        return df


def load_tier_one_sft_data(dataset_name_or_path: str = "FreedomIntelligence/medical-o1-reasoning-SFT"):
    """Loads the TIER-ONE (reasoning-style, non-MIMIC) SFT dataset.

    This is intentionally separate from everything in this file: tier-one
    data is used ONLY for supervised fine-tuning on reasoning style and must
    never be merged with tier-two (MIMIC-IV) records at the row/patient
    level. See Manuscript 3's plan for the rationale.
    """
    from datasets import load_dataset
    ds = load_dataset(dataset_name_or_path)
    return ds


if __name__ == "__main__":
    import argparse
    from mimic_loader import MimicConfig, MimicIVLoader, temporal_split

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_root", required=True)
    parser.add_argument("--cache_dir", default="./cache/mimic")
    parser.add_argument("--samples_per_stay", type=int, default=3)
    args = parser.parse_args()

    mcfg = MimicConfig(mimic_root=args.mimic_root, cache_dir=args.cache_dir)
    loader = MimicIVLoader(mcfg)
    cohort = loader.build_icu_cohort()
    deriv, evalu = temporal_split(cohort)

    vcfg = VignetteConfig()
    builder = VignetteBuilder(vcfg)

    # Scan chartevents/labevents ONCE for the full cohort, then split into
    # derivation/evaluation in memory. Scanning per-partition (the original
    # approach) streams the raw MIMIC-IV CSVs -- 40GB+ for chartevents alone
    # -- twice, since derivation and evaluation are disjoint subsets of the
    # same cohort; that doubled I/O time for no benefit and is what caused a
    # real 12-hour SLURM job timeout before this fix.
    logger.info("Extracting vitals/labs once for the full cohort (%d stays) ...", len(cohort))
    all_vitals = loader.extract_vitals_labs(cohort["stay_id"].tolist())
    all_labs = loader.extract_labs_for_admissions(cohort["hadm_id"].tolist())

    for split_name, split_cohort in [("derivation", deriv), ("evaluation", evalu)]:
        stay_id_set = set(split_cohort["stay_id"])
        hadm_id_set = set(split_cohort["hadm_id"])
        vitals = all_vitals[all_vitals["stay_id"].isin(stay_id_set)]
        labs = all_labs[all_labs["hadm_id"].isin(hadm_id_set)]
        vcfg.output_dir = f"./data/processed/{split_name}"
        Path(vcfg.output_dir).mkdir(parents=True, exist_ok=True)
        builder = VignetteBuilder(vcfg)
        builder.build_vignettes_for_cohort(split_cohort, vitals, labs,
                                            samples_per_stay=args.samples_per_stay)
