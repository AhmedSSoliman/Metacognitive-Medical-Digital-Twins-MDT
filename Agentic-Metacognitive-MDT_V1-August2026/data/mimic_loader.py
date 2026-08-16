"""
data/mimic_loader.py

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
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Build and cache the MIMIC-IV ICU cohort.")
    parser.add_argument("--mimic_root", required=True, help="Path to local MIMIC-IV release root")
    parser.add_argument("--cache_dir", default="./cache/mimic")
    args = parser.parse_args()

    cfg = MimicConfig(mimic_root=args.mimic_root, cache_dir=args.cache_dir)
    loader = MimicIVLoader(cfg)
    cohort = loader.build_icu_cohort()
    deriv, evalu = temporal_split(cohort)
    print(f"Cohort: {len(cohort)} stays | derivation: {len(deriv)} | evaluation: {len(evalu)}")
