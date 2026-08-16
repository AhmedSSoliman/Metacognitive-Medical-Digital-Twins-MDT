"""
core/cohort/terminology.py

Clinical terminology mapping utilities: ICD-9-CM / ICD-10-CM code prefixes
and the categorization + cohort-scope filtering built on them.

PORTING NOTE (2026-08-12): ported from the source repo's
data/cohort_stratification.py (../Agentic-DT_V1-July/data/cohort_stratification.py),
unchanged apart from this header and one import fix (the `_get_path` helper's
docstring reference to data/mimic_loader.py now points at core/cohort/mimic.py).

WHY THIS FILE AND NOT core/cohort/mimic.py: the target tree splits
"MIMIC-IV extraction, cohort stratification" (mimic.py) from "SNOMED-CT,
ICD-10, LOINC mapping utilities" (terminology.py). The source's
cohort_stratification.py is genuinely BOTH -- it defines the ICD-code
vocabulary (terminology mapping) AND applies it to filter cohorts (cohort
stratification), tightly interleaved: `categorize_icd_codes` is pure
terminology, while `apply_cohort_scope` is pure stratification, with
`categorize_admissions`/`filter_by_category` bridging them.

It was kept as ONE file here, placed under terminology.py, because:
  (a) the four functions form a single call chain (apply_cohort_scope ->
      categorize_admissions -> categorize_icd_codes; filter_by_category)
      and splitting them across two modules would create a circular-feeling
      dependency for no benefit;
  (b) the ICD prefix constants are the substantive, reviewable content here
      -- everything else is thin pandas glue over them -- so "terminology"
      is the better single home;
  (c) the source file's own docstring warns its ICD prefixes are duplicated
      in scripts/check_cohort_diagnostic_composition.py and must be changed
      in both places; adding a THIRD location by splitting would make that
      hazard worse.

NO SNOMED-CT AND NO LOINC MAPPING EXISTS. The target tree named all three
terminologies, but a grep for 'snomed' and 'loinc' across the entire source
repo returns hits only in this file's ICD code comments and its tests -- there
is no SNOMED-CT or LOINC code anywhere in the project. Lab and vital
identification is done by MIMIC-IV `itemid` integers instead (see
core/cohort/mimic.py's MimicConfig.vital_itemids / lab_itemids). If SNOMED-CT
or LOINC mapping is added later, this is the file for it.

ORIGINAL data/cohort_stratification.py MODULE DOCSTRING:

    Categorizes MIMIC-IV admissions by primary diagnosis (Cardiac / Sepsis /
    Both / Other) and provides the COHORT_SCOPE filter ("full" vs "cardiac")
    used by slurm/phase3_hypergraph.sbatch to control whether hyperedge mining
    runs over the full ICU cohort or only the subset with a cardiac-relevant
    diagnosis.

    This productionizes the categorization logic from the one-off
    scripts/check_cohort_diagnostic_composition.py (the standalone check that
    established the ~71.9% cardiac-relevant figure used in the AHA proposal
    framing, computed over the hemodynamic-instability triad cohort
    specifically) into a reusable module callable from the Phase 3 pipeline.
    That script intentionally stays self-contained (its own docstring: runnable
    without importing the rest of the project) so its ICD prefixes are
    duplicated here rather than imported -- if you change the cardiac/sepsis
    ICD prefix definitions, change both places.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# ICD-9-CM and ICD-10-CM code PREFIXES, matching
# scripts/check_cohort_diagnostic_composition.py. Prefix matching (not exact
# match) since ICD codes have many sub-codes (e.g. I21.0, I21.01, I21.02...
# are all "acute MI" variants).
CARDIAC_ICD10_PREFIXES = ["I21", "I22", "I23", "I24", "I25",  # ischemic heart disease / MI
                           "I50",                              # heart failure
                           "I46", "I47", "I48", "I49",         # cardiac arrest / arrhythmias
                           "R570"]                              # cardiogenic shock
CARDIAC_ICD9_PREFIXES = ["410", "411", "412", "413", "414",   # ischemic heart disease / MI
                          "428",                                # heart failure
                          "427",                                # arrhythmias
                          "7855"]                               # cardiogenic shock

SEPSIS_ICD10_PREFIXES = ["A40", "A41", "R6520", "R6521"]  # septicemia / sepsis / severe sepsis / septic shock
SEPSIS_ICD9_PREFIXES = ["038", "9959", "78552"]            # septicemia / SIRS-sepsis / septic shock

# Categories treated as "cardiac-relevant" for COHORT_SCOPE=cardiac filtering.
# "Both" (cardiac AND sepsis codes present) counts as cardiac-relevant since
# it does have a cardiac diagnosis, not despite it.
CARDIAC_RELEVANT_CATEGORIES = ("Cardiac", "Both")
VALID_SCOPES = ("full", "cardiac")


def _get_path(mimic_root: Path, parent_dir: str, stem: str) -> Path:
    """Same .csv.gz-with-.csv-fallback lookup as core/cohort/mimic.py's
    MimicIVLoader._get_path (was data/mimic_loader.py before this port).
    """
    gz_path = mimic_root / parent_dir / f"{stem}.csv.gz"
    if gz_path.exists():
        return gz_path
    return mimic_root / parent_dir / f"{stem}.csv"


def categorize_icd_codes(icd_codes: Iterable[str]) -> str:
    """Classifies a single admission's full list of billed ICD codes into
    one mutually-exclusive category: "Cardiac", "Sepsis", "Both", or
    "Other". An admission can have BOTH a cardiac AND a sepsis code (e.g.
    septic cardiomyopathy) -- reported as "Both" explicitly rather than
    silently picking one, since that ambiguity is itself clinically
    relevant (see check_cohort_diagnostic_composition.py's original
    rationale for this exact categorization).
    """
    codes = list(icd_codes)
    has_cardiac = any(
        any(code.startswith(p) for p in CARDIAC_ICD10_PREFIXES + CARDIAC_ICD9_PREFIXES)
        for code in codes
    )
    has_sepsis = any(
        any(code.startswith(p) for p in SEPSIS_ICD10_PREFIXES + SEPSIS_ICD9_PREFIXES)
        for code in codes
    )
    if has_cardiac and has_sepsis:
        return "Both"
    elif has_cardiac:
        return "Cardiac"
    elif has_sepsis:
        return "Sepsis"
    else:
        return "Other"


def categorize_admissions(mimic_root: str, hadm_ids: Iterable[int]) -> pd.Series:
    """Loads diagnoses_icd for the given hadm_ids and returns a Series
    indexed by hadm_id, valued by category ("Cardiac"/"Sepsis"/"Both"/
    "Other"). Admissions with zero rows in diagnoses_icd are categorized
    "Other" -- a conservative default, since absence of a billed cardiac or
    sepsis code is not evidence an admission belongs in either category.
    """
    hadm_id_set = {int(h) for h in hadm_ids}
    if not hadm_id_set:
        return pd.Series(dtype=object)

    diag_path = _get_path(Path(mimic_root), "hosp", "diagnoses_icd")
    diagnoses = pd.read_csv(diag_path, usecols=["hadm_id", "icd_code"], dtype={"icd_code": str})
    diagnoses = diagnoses.dropna(subset=["hadm_id"])
    diagnoses["hadm_id"] = diagnoses["hadm_id"].astype(int)
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(hadm_id_set)]

    codes_by_admission = diagnoses.groupby("hadm_id")["icd_code"].apply(list)
    categories = codes_by_admission.apply(categorize_icd_codes)

    missing = hadm_id_set - set(categories.index)
    if missing:
        logger.warning(
            "%d admission(s) had no rows in diagnoses_icd; categorizing as 'Other'.",
            len(missing),
        )
        categories = pd.concat([categories, pd.Series({h: "Other" for h in missing})])

    return categories.reindex(sorted(hadm_id_set))


def filter_by_category(cohort: pd.DataFrame, categories: pd.Series,
                        keep: Iterable[str] = CARDIAC_RELEVANT_CATEGORIES,
                        hadm_id_col: str = "hadm_id") -> pd.DataFrame:
    """Filters a cohort DataFrame down to admissions whose category (looked
    up per-row in `categories`, indexed by hadm_id) is in `keep`.

    Applying this to the derivation and evaluation partitions SEPARATELY,
    each against the same `categories` Series and `keep` set, is what makes
    COHORT_SCOPE=cardiac filter both partitions identically -- there is no
    partition-specific branching anywhere in this function.
    """
    keep = set(keep)
    mask = cohort[hadm_id_col].map(categories).isin(keep)
    return cohort[mask].copy()


def apply_cohort_scope(mimic_root: str, cohort: pd.DataFrame, scope: str,
                        hadm_id_col: str = "hadm_id") -> pd.DataFrame:
    """Top-level entry point used by Phase 3 (slurm/phase3_hypergraph.sbatch):
    given a cohort DataFrame (e.g. a derivation or evaluation partition from
    data.mimic_loader.temporal_split) and a COHORT_SCOPE value ("full" or
    "cardiac"), returns the appropriately filtered cohort.

    "full" returns the cohort unchanged -- no diagnoses_icd read needed, so
    the default (unset COHORT_SCOPE) path has zero added cost.
    """
    if scope not in VALID_SCOPES:
        raise ValueError(f"Unknown COHORT_SCOPE '{scope}' -- expected one of {VALID_SCOPES}")
    if scope == "full":
        return cohort
    categories = categorize_admissions(mimic_root, cohort[hadm_id_col].tolist())
    return filter_by_category(cohort, categories, keep=CARDIAC_RELEVANT_CATEGORIES, hadm_id_col=hadm_id_col)
