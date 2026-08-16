"""
check_cohort_diagnostic_composition.py

Answers the exact question flagged before writing any more of the AHA
proposal: of the admissions exhibiting the hemodynamic-instability triad
(tachycardia + hypotension + hyperlactatemia) that the hypergraph is built
around, what fraction have a PRIMARY CARDIAC diagnosis (MI, heart failure,
arrhythmia, cardiogenic shock) versus sepsis versus something else entirely?

This determines whether the AHA cardiovascular framing is honestly
supportable by this specific cohort, or whether it needs a different cohort
definition (e.g. filtering to admissions with a cardiac diagnosis present)
before submission.

Uses the same itemids already established in data/mimic_loader.py's
MimicConfig for consistency with the rest of the project's pipeline.

USAGE:
    python check_cohort_diagnostic_composition.py --mimic_root /path/to/mimic-iv-3.1
"""

import argparse
from pathlib import Path

import pandas as pd

# Same itemids as data/mimic_loader.py's MimicConfig, repeated here so this
# script is self-contained and runnable without importing the rest of the project.
HEART_RATE_ITEMS = [220045]
MAP_ITEMS = [220052, 220181, 225312]
LACTATE_ITEMS = [50813]

TACHYCARDIA_THRESHOLD = 100.0   # bpm
HYPOTENSION_THRESHOLD = 65.0    # mmHg (MAP)
HYPERLACTATEMIA_THRESHOLD = 2.0  # mmol/L

# ICD-9-CM and ICD-10-CM code PREFIXES for the categories being checked.
# Using startswith() prefix matching (not exact match) since ICD codes have
# many sub-codes (e.g. I21.0, I21.01, I21.02... are all "acute MI" variants).
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


def _get_path(mimic_root: Path, parent_dir: str, stem: str) -> Path:
    """Same .csv.gz-with-.csv-fallback lookup as data/mimic_loader.py's
    MimicIVLoader._get_path, repeated here since this script is meant to be
    self-contained. MIMIC-IV releases are distributed as .csv.gz but some
    local copies (including this project's) are stored already decompressed.
    """
    gz_path = mimic_root / parent_dir / f"{stem}.csv.gz"
    if gz_path.exists():
        return gz_path
    csv_path = mimic_root / parent_dir / f"{stem}.csv"
    if csv_path.exists():
        return csv_path
    return gz_path


def categorize_icd_codes(icd_codes: list[str]) -> str:
    """Classifies a single admission's full list of billed ICD codes into
    one mutually-informative category. An admission can have BOTH a cardiac
    AND a sepsis code (e.g. septic cardiomyopathy) -- reported as "Both"
    explicitly rather than silently picking one, since that ambiguity is
    itself relevant to how defensible a pure-cardiovascular framing is.
    """
    has_cardiac = any(
        any(code.startswith(p) for p in CARDIAC_ICD10_PREFIXES + CARDIAC_ICD9_PREFIXES)
        for code in icd_codes
    )
    has_sepsis = any(
        any(code.startswith(p) for p in SEPSIS_ICD10_PREFIXES + SEPSIS_ICD9_PREFIXES)
        for code in icd_codes
    )
    if has_cardiac and has_sepsis:
        return "Both cardiac and sepsis codes present"
    elif has_cardiac:
        return "Cardiac"
    elif has_sepsis:
        return "Sepsis"
    else:
        return "Other (neither cardiac nor sepsis code present)"


def find_triad_admissions(mimic_root: Path, chunksize: int = 5_000_000) -> set[int]:
    """Streams chartevents + labevents to find hadm_ids where tachycardia,
    hypotension, and hyperlactatemia ALL occurred at some point during the
    admission (not necessarily simultaneously -- a looser first-pass filter;
    tighten to a specific co-occurrence window if you want to match the
    hypergraph's actual 6h-window definition exactly).
    """
    print("Scanning chartevents for tachycardia + hypotension ...")
    hr_hadm_ids = set()
    map_hadm_ids = set()

    for chunk in pd.read_csv(
        _get_path(mimic_root, "icu", "chartevents"),
        usecols=["hadm_id", "itemid", "valuenum"],
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        hr_chunk = chunk[chunk["itemid"].isin(HEART_RATE_ITEMS) & (chunk["valuenum"] > TACHYCARDIA_THRESHOLD)]
        hr_hadm_ids.update(hr_chunk["hadm_id"].astype(int).unique())

        map_chunk = chunk[chunk["itemid"].isin(MAP_ITEMS) & (chunk["valuenum"] < HYPOTENSION_THRESHOLD)]
        map_hadm_ids.update(map_chunk["hadm_id"].astype(int).unique())

    print(f"  Admissions with tachycardia at some point: {len(hr_hadm_ids)}")
    print(f"  Admissions with hypotension at some point: {len(map_hadm_ids)}")

    print("Scanning labevents for hyperlactatemia ...")
    lactate_hadm_ids = set()
    for chunk in pd.read_csv(
        _get_path(mimic_root, "hosp", "labevents"),
        usecols=["hadm_id", "itemid", "valuenum"],
        chunksize=chunksize,
    ):
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        lac_chunk = chunk[chunk["itemid"].isin(LACTATE_ITEMS) & (chunk["valuenum"] > HYPERLACTATEMIA_THRESHOLD)]
        lactate_hadm_ids.update(lac_chunk["hadm_id"].unique())

    print(f"  Admissions with hyperlactatemia at some point: {len(lactate_hadm_ids)}")

    triad_hadm_ids = hr_hadm_ids & map_hadm_ids & lactate_hadm_ids
    print(f"\nAdmissions with ALL THREE (the triad cohort): {len(triad_hadm_ids)}")
    return triad_hadm_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_root", required=True, help="Path to local MIMIC-IV release root")
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    args = parser.parse_args()

    mimic_root = Path(args.mimic_root)

    triad_hadm_ids = find_triad_admissions(mimic_root, chunksize=args.chunksize)
    if not triad_hadm_ids:
        print("No admissions found matching the triad -- check itemids/thresholds against your MIMIC-IV release.")
        return

    print("\nLoading diagnoses_icd for the triad cohort ...")
    diagnoses = pd.read_csv(
        _get_path(mimic_root, "hosp", "diagnoses_icd"),
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    diagnoses = diagnoses[diagnoses["hadm_id"].isin(triad_hadm_ids)]
    print(f"  {len(diagnoses)} diagnosis rows found across {diagnoses['hadm_id'].nunique()} of the triad admissions")

    codes_by_admission = diagnoses.groupby("hadm_id")["icd_code"].apply(list)

    print("\nCategorizing each admission ...")
    categories = codes_by_admission.apply(categorize_icd_codes)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPOSITION OF THE HEMODYNAMIC-INSTABILITY TRIAD COHORT")
    print("=" * 60)
    counts = categories.value_counts()
    pcts = categories.value_counts(normalize=True) * 100
    for cat in counts.index:
        print(f"  {cat:<45} {counts[cat]:>6}  ({pcts[cat]:.1f}%)")
    print(f"  {'TOTAL':<45} {len(categories):>6}  (100.0%)")

    cardiac_relevant_pct = pcts.get("Cardiac", 0) + pcts.get("Both cardiac and sepsis codes present", 0)
    print(f"\n>>> Admissions with ANY cardiac-relevant code: {cardiac_relevant_pct:.1f}% <<<")
    print(
        "\nInterpretation guide: if this figure is low (e.g. well under 30-40%), the "
        "AHA cardiovascular framing as currently scoped (the general triad, unfiltered "
        "by diagnosis) is not well supported by this specific cohort -- consider either "
        "(a) restricting the cohort to admissions with a cardiac diagnosis present before "
        "computing the hypergraph, narrowing the clinical claim to match, or (b) reconsidering "
        "AHA in favor of a funding mechanism without a cardiovascular-specific scope requirement."
    )


if __name__ == "__main__":
    main()
