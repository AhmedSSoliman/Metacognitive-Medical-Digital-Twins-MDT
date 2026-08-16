"""
core/cohort/mimic.py

MIMIC-IV extraction, cohort construction, temporal partitioning, and the
vignette/label preprocessing built on top of it.

PORTING NOTE (2026-08-12): this file merges TWO source modules, both of which
are "extraction/labelling from MIMIC-IV" rather than terminology mapping:
  - ../Agentic-DT_V1-July/data/mimic_loader.py  (MimicConfig, MimicIVLoader,
    temporal_split) -- Part 1 below.
  - ../Agentic-DT_V1-July/data/preprocessing.py (VignetteConfig,
    VignetteBuilder, load_tier_one_sft_data) -- Part 2 below.
They were merged because preprocessing.py's whole job is turning what
mimic_loader.py extracts into vignettes + deterioration labels; they were
already coupled (preprocessing.py's __main__ imports mimic_loader directly)
and the target tree allocates one file, core/cohort/mimic.py, for "MIMIC-IV
extraction, cohort stratification".

The ICD-code CATEGORIZATION logic (the "cardiac vs sepsis vs other"
stratification from data/cohort_stratification.py) went to
core/cohort/terminology.py instead -- see that file's own porting note for
the reasoning on that split.

Both parts are byte-identical to their sources apart from: this header, the
two part banners, one merged import block, and the __main__ block (which is
now a single combined block, since two `if __name__ == "__main__":` blocks in
one file would shadow each other -- mimic_loader.py's cohort-build CLI and
preprocessing.py's vignette-build CLI are exposed as two subcommands).

ORIGINAL data/mimic_loader.py MODULE DOCSTRING:

    MIMIC-IV data loading and cohort construction for the Metacognitive Medical
    Digital Twin (MDT) project.

    Assumes the user has local, credentialed access to MIMIC-IV (v2.2+) and
    MIMIC-IV-Note, downloaded per PhysioNet's data use agreement. This module
    does NOT fetch or redistribute any data -- it only reads from a local path
    you provide via config.

    Expected directory layout (standard MIMIC-IV release structure):
        <mimic_root>/hosp/patients.csv.gz
        <mimic_root>/hosp/admissions.csv.gz
        <mimic_root>/hosp/labevents.csv.gz
        <mimic_root>/hosp/d_labitems.csv.gz
        <mimic_root>/hosp/prescriptions.csv.gz
        <mimic_root>/icu/icustays.csv.gz
        <mimic_root>/icu/chartevents.csv.gz
        <mimic_root>/icu/d_items.csv.gz
        <mimic_root>/note/discharge.csv.gz         (MIMIC-IV-Note)
        <mimic_root>/note/radiology.csv.gz         (MIMIC-IV-Note)

ORIGINAL data/preprocessing.py MODULE DOCSTRING:

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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# PART 1 -- MIMIC-IV loading, cohort construction, temporal split
# (source: data/mimic_loader.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MimicConfig:
    mimic_root: str  # path to local MIMIC-IV root, e.g. "/blue/bihorac/shared/mimic-iv-2.2"
    min_icu_los_hours: float = 6.0          # exclude ultra-short ICU stays
    max_icu_los_days: float = 30.0          # exclude extreme outliers
    min_age: int = 18
    cache_dir: str = "./cache/mimic"
    # Vital sign itemids commonly used at the bedside (MIMIC-IV chartevents d_items)
    # NOTE: verify these against your own d_items extract before large runs --
    # itemid mappings can differ slightly across MIMIC-IV point releases.
    vital_itemids: dict = field(default_factory=lambda: {
        "heart_rate": [220045],
        "sbp": [220050, 220179],
        "dbp": [220051, 220180],
        "map": [220052, 220181, 225312],
        "resp_rate": [220210],
        "spo2": [220277],
        "temperature_c": [223762],
        "gcs_total": [220739],  # if using a precomputed total; else sum motor/verbal/eye
    })
    lab_itemids: dict = field(default_factory=lambda: {
        "lactate": [50813],
        "creatinine": [50912],
        "wbc": [51301],
        "platelet": [51265],
        "bilirubin_total": [50885],
        "hemoglobin": [51222],
        "sodium": [50983],
        "potassium": [50971],
        "ph_arterial": [50820],
    })


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

class MimicIVLoader:
    """Reads and joins the subset of MIMIC-IV tables needed for the MDT
    pipeline: cohort definition, vitals/labs time series, and clinical notes.
    """

    def __init__(self, cfg: MimicConfig):
        self.cfg = cfg
        self.root = Path(cfg.mimic_root)
        os.makedirs(cfg.cache_dir, exist_ok=True)
        self._validate_paths()

    def _get_path(self, parent_dir: str, stem: str) -> Path:
        gz_path = self.root / parent_dir / f"{stem}.csv.gz"
        if gz_path.exists():
            return gz_path
        csv_path = self.root / parent_dir / f"{stem}.csv"
        if csv_path.exists():
            return csv_path
        return gz_path

    def _validate_paths(self):
        required = [
            self._get_path("hosp", "patients"),
            self._get_path("hosp", "admissions"),
            self._get_path("icu", "icustays"),
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"MIMIC-IV root '{self.root}' is missing expected files: {missing}. "
                f"Check that MimicConfig.mimic_root points at the correct release directory."
            )

    # -- Cohort construction -------------------------------------------------

    def build_icu_cohort(self) -> pd.DataFrame:
        """Returns one row per ICU stay meeting inclusion criteria, with
        demographics and admission context joined in.
        """
        cache_path = Path(self.cfg.cache_dir) / "icu_cohort.parquet"
        if cache_path.exists():
            logger.info("Loading cached ICU cohort from %s", cache_path)
            return pd.read_parquet(cache_path)

        logger.info("Building ICU cohort from raw MIMIC-IV tables ...")
        patients = pd.read_csv(self._get_path("hosp", "patients"),
                                usecols=["subject_id", "gender", "anchor_age", "anchor_year", "dod"])
        admissions = pd.read_csv(self._get_path("hosp", "admissions"),
                                  usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                                           "deathtime", "admission_type", "insurance",
                                           "race", "hospital_expire_flag"],
                                  parse_dates=["admittime", "dischtime", "deathtime"])
        icustays = pd.read_csv(self._get_path("icu", "icustays"),
                                parse_dates=["intime", "outtime"])

        cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="left")
        cohort = cohort.merge(patients, on="subject_id", how="left")

        cohort["age_at_admission"] = cohort["anchor_age"] + (
            cohort["admittime"].dt.year - cohort["anchor_year"]
        )
        cohort = cohort[cohort["age_at_admission"] >= self.cfg.min_age]

        cohort["icu_los_hours"] = (
            cohort["outtime"] - cohort["intime"]
        ).dt.total_seconds() / 3600.0
        cohort = cohort[
            (cohort["icu_los_hours"] >= self.cfg.min_icu_los_hours)
            & (cohort["icu_los_hours"] <= self.cfg.max_icu_los_days * 24.0)
        ]

        # Mortality labels: in-hospital and ICU-window death
        cohort["mortality_inhospital"] = cohort["hospital_expire_flag"].fillna(0).astype(int)
        cohort["died_in_icu_window"] = (
            cohort["deathtime"].notna()
            & (cohort["deathtime"] >= cohort["intime"])
            & (cohort["deathtime"] <= cohort["outtime"] + pd.Timedelta(hours=24))
        ).astype(int)

        cohort = cohort.reset_index(drop=True)
        cohort.to_parquet(cache_path)
        logger.info("Built cohort with %d ICU stays", len(cohort))
        return cohort

    # -- Time-series extraction ----------------------------------------------

    def extract_vitals_labs(self, stay_ids: list[int], chunksize: int = 5_000_000) -> pd.DataFrame:
        """Streams chartevents + labevents for the given ICU stay IDs and
        returns a long-format time series: [stay_id, charttime, variable, value].

        Uses chunked reading since chartevents is typically >100GB uncompressed.
        """
        stay_id_set = set(stay_ids)
        all_vital_ids = [i for ids in self.cfg.vital_itemids.values() for i in ids]
        vital_id_to_name = {i: name for name, ids in self.cfg.vital_itemids.items() for i in ids}

        frames = []
        chart_path = self._get_path("icu", "chartevents")
        logger.info("Streaming chartevents from %s (this may take a while) ...", chart_path)
        for chunk in pd.read_csv(
            chart_path,
            usecols=["stay_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"],
            chunksize=chunksize,
        ):
            chunk = chunk[chunk["stay_id"].isin(stay_id_set) & chunk["itemid"].isin(all_vital_ids)]
            if len(chunk):
                chunk["variable"] = chunk["itemid"].map(vital_id_to_name)
                frames.append(chunk[["stay_id", "charttime", "variable", "valuenum"]]
                              .rename(columns={"valuenum": "value"}))

        vitals = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["stay_id", "charttime", "variable", "value"]
        )

        # Labs are joined via hadm_id, not stay_id -- caller must have hadm_id mapping
        # available; see extract_labs_for_admissions for the hadm_id-keyed version.
        return vitals

    def extract_labs_for_admissions(self, hadm_ids: list[int], chunksize: int = 2_000_000) -> pd.DataFrame:
        hadm_id_set = set(hadm_ids)
        all_lab_ids = [i for ids in self.cfg.lab_itemids.values() for i in ids]
        lab_id_to_name = {i: name for name, ids in self.cfg.lab_itemids.items() for i in ids}

        frames = []
        lab_path = self._get_path("hosp", "labevents")
        logger.info("Streaming labevents from %s ...", lab_path)
        for chunk in pd.read_csv(
            lab_path,
            usecols=["hadm_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"],
            chunksize=chunksize,
        ):
            chunk = chunk.dropna(subset=["hadm_id"])
            chunk["hadm_id"] = chunk["hadm_id"].astype(int)
            chunk = chunk[chunk["hadm_id"].isin(hadm_id_set) & chunk["itemid"].isin(all_lab_ids)]
            if len(chunk):
                chunk["variable"] = chunk["itemid"].map(lab_id_to_name)
                frames.append(chunk[["hadm_id", "charttime", "variable", "valuenum"]]
                              .rename(columns={"valuenum": "value"}))

        labs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["hadm_id", "charttime", "variable", "value"]
        )
        return labs

    # -- Notes ----------------------------------------------------------------

    def load_discharge_notes(self, hadm_ids: Optional[list[int]] = None) -> pd.DataFrame:
        """Loads MIMIC-IV-Note discharge summaries. Requires separate
        PhysioNet credentialing for the -Note module beyond base MIMIC-IV.
        """
        note_path = self._get_path("note", "discharge")
        if not note_path.exists():
            raise FileNotFoundError(
                f"{note_path} not found. MIMIC-IV-Note requires a separate, additional "
                f"PhysioNet data use agreement beyond base MIMIC-IV -- confirm you have "
                f"downloaded and placed it under <mimic_root>/note/."
            )
        notes = pd.read_csv(note_path, usecols=["note_id", "hadm_id", "charttime", "text"])
        if hadm_ids is not None:
            notes = notes[notes["hadm_id"].isin(set(hadm_ids))]
        return notes



# ---------------------------------------------------------------------------
# Temporal split (CRITICAL for Aim 3 hypergraph derivation/evaluation separation)
# ---------------------------------------------------------------------------

def temporal_split(cohort: pd.DataFrame, admittime_col: str = "admittime",
                    derivation_frac: float = 0.6, seed: int = 42):
    """Splits the cohort by admission date into a derivation partition (used
    ONLY for hyperedge mining, Aim 3) and an evaluation partition (used ONLY
    for held-out metrics), so the two never overlap on the same admission.

    This is a hard requirement from the proposal's data-governance design --
    do not replace with a random split.
    """
    cohort_sorted = cohort.sort_values(admittime_col).reset_index(drop=True)
    cutoff_idx = int(len(cohort_sorted) * derivation_frac)
    cutoff_date = cohort_sorted.iloc[cutoff_idx][admittime_col]

    derivation = cohort_sorted[cohort_sorted[admittime_col] < cutoff_date].copy()
    evaluation = cohort_sorted[cohort_sorted[admittime_col] >= cutoff_date].copy()

    logger.info(
        "Temporal split at %s: derivation n=%d, evaluation n=%d",
        cutoff_date, len(derivation), len(evaluation),
    )
    assert set(derivation["hadm_id"]).isdisjoint(set(evaluation["hadm_id"])), \
        "Temporal split leaked admissions across partitions -- this must never happen."
    return derivation, evaluation




# ===========================================================================
# PART 2 -- vignette construction and deterioration labelling
# (source: data/preprocessing.py)
# ===========================================================================

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
    # Combined CLI for the two merged source modules. Previously these were
    # two separate `python data/mimic_loader.py ...` / `python
    # data/preprocessing.py ...` entry points; merging the modules means one
    # __main__ with a subcommand selector, so both original behaviors remain
    # reachable and unchanged.
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="MIMIC-IV cohort build / vignette build.")
    parser.add_argument("command", choices=["build-cohort", "build-vignettes"],
                        help="build-cohort: original data/mimic_loader.py __main__. "
                             "build-vignettes: original data/preprocessing.py __main__.")
    parser.add_argument("--mimic_root", required=True, help="Path to local MIMIC-IV release root")
    parser.add_argument("--cache_dir", default="./cache/mimic")
    parser.add_argument("--samples_per_stay", type=int, default=3)
    args = parser.parse_args()

    if args.command == "build-cohort":
        # --- original data/mimic_loader.py __main__, unchanged ---
        cfg = MimicConfig(mimic_root=args.mimic_root, cache_dir=args.cache_dir)
        loader = MimicIVLoader(cfg)
        cohort = loader.build_icu_cohort()
        deriv, evalu = temporal_split(cohort)
        print(f"Cohort: {len(cohort)} stays | derivation: {len(deriv)} | evaluation: {len(evalu)}")
    else:
        # --- original data/preprocessing.py __main__, unchanged apart from
        # dropping its `from mimic_loader import ...` line (those names are
        # now defined above in this same module). ---
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
