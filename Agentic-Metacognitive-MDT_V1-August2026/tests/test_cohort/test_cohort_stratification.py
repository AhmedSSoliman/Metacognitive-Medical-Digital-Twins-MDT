"""
tests/test_cohort/test_cohort_stratification.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_cohort_stratification.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Synthetic-data tests for core/cohort/terminology.py's ICD categorization
and COHORT_SCOPE filtering logic. No torch dependency -- pandas only. See
that module's docstring for why the ICD prefix lists are duplicated from
scripts/check_cohort_diagnostic_composition.py rather than imported from it.

NOTE: these tests exercise the categorization/filtering LOGIC against
synthetic ICD codes and cohorts only. They do not touch real MIMIC-IV data
-- run scripts/check_cohort_diagnostic_composition.py (or
slurm/check_cohort_composition.sbatch) against a real MIMIC-IV release to
verify the ~71.9% cardiac-relevant figure reproduces; that is a data
verification question, not something a unit test can answer.
"""

import pandas as pd
import pytest
import sys
from pathlib import Path

from core.cohort.terminology import (
    categorize_icd_codes, categorize_admissions, filter_by_category,
    apply_cohort_scope, CARDIAC_RELEVANT_CATEGORIES,
)


# ---------------------------------------------------------------------------
# categorize_icd_codes
# ---------------------------------------------------------------------------

def test_categorize_pure_cardiac_icd10():
    assert categorize_icd_codes(["I2101", "E119"]) == "Cardiac"


def test_categorize_pure_cardiac_icd9():
    assert categorize_icd_codes(["41071", "2724"]) == "Cardiac"


def test_categorize_pure_sepsis():
    assert categorize_icd_codes(["A419", "R6521"]) == "Sepsis"


def test_categorize_both_cardiac_and_sepsis():
    # Septic cardiomyopathy-style admission: both a cardiac and sepsis code present.
    assert categorize_icd_codes(["I50 9", "A419"]) == "Both"


def test_categorize_other_when_neither_present():
    assert categorize_icd_codes(["J189", "N179"]) == "Other"


def test_categorize_other_for_empty_code_list():
    assert categorize_icd_codes([]) == "Other"


def test_categorize_uses_prefix_not_exact_match():
    # I21 is "acute MI" -- sub-codes like I2109 must also match via prefix.
    assert categorize_icd_codes(["I2109"]) == "Cardiac"


# ---------------------------------------------------------------------------
# categorize_admissions (against a synthetic diagnoses_icd table)
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_mimic_root(tmp_path):
    """Builds a minimal synthetic <mimic_root>/hosp/diagnoses_icd.csv on
    disk so categorize_admissions can be exercised through its real file-
    reading path without needing a real MIMIC-IV release.
    """
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir()
    diagnoses = pd.DataFrame({
        "hadm_id": [1, 1, 2, 3, 3, 4],
        "icd_code": ["I2101", "E119", "A419", "I50 9", "A419", "J189"],
    })
    diagnoses.to_csv(hosp_dir / "diagnoses_icd.csv", index=False)
    return tmp_path


def test_categorize_admissions_matches_per_admission_categories(synthetic_mimic_root):
    categories = categorize_admissions(str(synthetic_mimic_root), [1, 2, 3, 4])
    assert categories.loc[1] == "Cardiac"
    assert categories.loc[2] == "Sepsis"
    assert categories.loc[3] == "Both"
    assert categories.loc[4] == "Other"


def test_categorize_admissions_defaults_missing_hadm_id_to_other(synthetic_mimic_root):
    # hadm_id 999 has no rows in diagnoses_icd at all.
    categories = categorize_admissions(str(synthetic_mimic_root), [1, 999])
    assert categories.loc[999] == "Other"


def test_categorize_admissions_empty_input_returns_empty_series(synthetic_mimic_root):
    categories = categorize_admissions(str(synthetic_mimic_root), [])
    assert len(categories) == 0


# ---------------------------------------------------------------------------
# filter_by_category / apply_cohort_scope
# ---------------------------------------------------------------------------

def _synthetic_cohort(hadm_ids):
    return pd.DataFrame({"hadm_id": hadm_ids, "some_other_col": range(len(hadm_ids))})


def test_filter_by_category_keeps_only_cardiac_relevant():
    cohort = _synthetic_cohort([1, 2, 3, 4])
    categories = pd.Series({1: "Cardiac", 2: "Sepsis", 3: "Both", 4: "Other"})
    filtered = filter_by_category(cohort, categories, keep=CARDIAC_RELEVANT_CATEGORIES)
    assert sorted(filtered["hadm_id"]) == [1, 3]


def test_filter_by_category_applies_identically_to_multiple_cohorts():
    # The whole point of COHORT_SCOPE=cardiac is that derivation and
    # evaluation partitions are filtered by the SAME categorization logic,
    # independently -- neither partition should get special-cased treatment.
    categories = pd.Series({1: "Cardiac", 2: "Sepsis", 3: "Both", 4: "Other", 5: "Cardiac"})

    derivation = _synthetic_cohort([1, 2, 3])
    evaluation = _synthetic_cohort([4, 5])

    filtered_derivation = filter_by_category(derivation, categories, keep=CARDIAC_RELEVANT_CATEGORIES)
    filtered_evaluation = filter_by_category(evaluation, categories, keep=CARDIAC_RELEVANT_CATEGORIES)

    assert sorted(filtered_derivation["hadm_id"]) == [1, 3]
    assert sorted(filtered_evaluation["hadm_id"]) == [5]

    # No admission should ever appear filtered-in on one partition and
    # filtered-out on the other for the same category -- verify by applying
    # the identical filter to the concatenation and checking it matches the
    # union of the two partition-level results exactly.
    combined = pd.concat([derivation, evaluation], ignore_index=True)
    filtered_combined = filter_by_category(combined, categories, keep=CARDIAC_RELEVANT_CATEGORIES)
    expected_union = sorted(list(filtered_derivation["hadm_id"]) + list(filtered_evaluation["hadm_id"]))
    assert sorted(filtered_combined["hadm_id"]) == expected_union


def test_apply_cohort_scope_full_returns_cohort_unchanged(synthetic_mimic_root):
    cohort = _synthetic_cohort([1, 2, 3, 4])
    result = apply_cohort_scope(str(synthetic_mimic_root), cohort, "full")
    assert sorted(result["hadm_id"]) == [1, 2, 3, 4]


def test_apply_cohort_scope_cardiac_filters_via_real_lookup(synthetic_mimic_root):
    cohort = _synthetic_cohort([1, 2, 3, 4])
    result = apply_cohort_scope(str(synthetic_mimic_root), cohort, "cardiac")
    # Admission 1 = Cardiac, 3 = Both -> both cardiac-relevant; 2 = Sepsis, 4 = Other -> excluded.
    assert sorted(result["hadm_id"]) == [1, 3]


def test_apply_cohort_scope_rejects_unknown_scope(synthetic_mimic_root):
    cohort = _synthetic_cohort([1, 2])
    with pytest.raises(ValueError):
        apply_cohort_scope(str(synthetic_mimic_root), cohort, "not_a_real_scope")
