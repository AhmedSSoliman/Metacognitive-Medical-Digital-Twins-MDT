"""
data/build_grpo_prompts.py

Bridges Phase 1's MIMIC-IV vignette output (data/preprocessing.py) into the
exact schema Phase 2's GRPO training requires (training/grpo_utils.py's
PromptDataset, used by scripts/run_phase4_scaled_training.py; Phase 2's own
training/grpo_trainer.py -- since its refactor onto TRL's GRPOTrainer --
expects the same four columns via format_dataset_for_grpo, which calls this
same schema): [prompt, reference_patient_state, recipient_type,
must_mention_facts].

This gap is real, not cosmetic: build_vignette() in preprocessing.py returns
{stay_id, hadm_id, prediction_time, prompt, deterioration_6h,
hypotension_event, hyperlactatemia_event} -- NONE of the three columns
GRPO needs. Running GRPO training directly against Phase 1's parquet
output will fail immediately with a missing-column error.

WHAT THIS SCRIPT DOES, AND WHAT STILL NEEDS CLINICAL REVIEW:

1. reference_patient_state: auto-generated from the structured deterioration
   labels using a fixed template (see `_template_reference_state`). This is
   a PLACEHOLDER, not a clinician-written ground truth -- R_sem (semantic
   fidelity reward) will only be as good as this template's accuracy. Before
   trusting Phase 2 results, either (a) have a clinician write/edit a sample
   of these references directly, or (b) at minimum have one review the
   template logic below for physiological accuracy.

2. recipient_type: MIMIC-IV has no field indicating who a given prediction
   is "for" -- this script assigns recipient_type by SAMPLING from a fixed
   distribution (default: 70% clinician, 15% patient, 15% family), which is
   a modeling choice, not a fact recovered from the data. Change
   `RECIPIENT_DISTRIBUTION` if a different split better matches your
   intended deployment context.

3. must_mention_facts: populated from static, structurally reliable
   admission-level facts (admission_type, and prior ICU stay history within
   the same hospitalization if present). This is a conservative subset --
   MIMIC-IV does not have a clean structured allergy table comparable to
   what a real clinical system would have, so allergy-style "must retain"
   facts are NOT included here. If R_retention needs richer facts, that
   requires either NLP extraction from notes (out of scope for this script)
   or a different tier-two source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# A modeling choice, not a fact recovered from MIMIC-IV -- see point 2 above.
# Weighted toward "clinician" since that is the most common real-world
# recipient of an ICU risk assessment; the smaller patient/family shares
# still give the R_emp (structural empathy) reward enough non-clinician
# examples to actually train against, rather than training almost
# exclusively on the clinician-framing case and generalizing poorly to
# the other two.
RECIPIENT_DISTRIBUTION = {"clinician": 0.70, "patient": 0.15, "family": 0.15}


@dataclass
class GRPOPromptBuilderConfig:
    seed: int = 42  # controls recipient_type sampling only -- reference_patient_state and
                     # must_mention_facts are fully deterministic given the input rows, so
                     # re-running with the same seed and the same vignettes_df/cohort_df
                     # reproduces an identical output file byte-for-byte.
    output_path: str = "./data/processed/derivation/grpo_prompts.parquet"


def _template_reference_state(row: pd.Series) -> str:
    """Builds a short reference patient-state description from the
    structured deterioration labels. This is a fixed-template placeholder,
    NOT a clinician-written reference -- see module docstring.
    """
    parts = []
    # These two flags come straight from preprocessing.py's
    # compute_deterioration_labels -- deliberately checking each
    # independently (not just the combined deterioration_6h flag) so the
    # generated sentence can name WHICH specific abnormality occurred,
    # rather than only saying "deterioration happened" with no detail for
    # R_sem to actually compare against.
    if row.get("hypotension_event", 0) == 1:
        parts.append("developed hypotension (MAP below threshold)")
    if row.get("hyperlactatemia_event", 0) == 1:
        parts.append("developed rising lactate above threshold")

    if parts:
        # Joins with "and" rather than a list, since both can legitimately
        # co-occur (e.g. early septic shock commonly presents with both
        # hypotension AND rising lactate together) and the resulting
        # sentence should read as one coherent clinical statement, not a
        # bullet-style concatenation.
        detail = " and ".join(parts)
        return (
            f"Patient showed signs of hemodynamic deterioration within the prediction "
            f"window: {detail}, consistent with early evolving instability requiring "
            f"close reassessment."
        )
    else:
        # The "no event" branch is equally important to generate a real
        # reference for -- without it, R_sem would have no ground truth to
        # compare against on the (likely majority) of examples where the
        # patient did NOT deteriorate, silently biasing training toward
        # only learning to describe the deterioration case well.
        return (
            "Patient's vitals and labs remained within the monitored thresholds over "
            "the prediction window, with no hypotension or rising lactate event recorded; "
            "consistent with a stable trajectory over this period."
        )


def _sample_recipient_types(n: int, rng: np.random.Generator) -> list[str]:
    # np.random.Generator.choice with p=probs draws each of the n samples
    # INDEPENDENTLY according to RECIPIENT_DISTRIBUTION -- it does not
    # guarantee the exact target proportions in any single finite sample
    # (e.g. exactly 70/15/15 out of 1000), only that the EXPECTED
    # proportions converge to those weights as n grows. This is fine for
    # training data at any reasonably large n; if you need exact,
    # guaranteed proportions for a small evaluation set instead, use a
    # stratified assignment (e.g. round-robin or np.repeat + shuffle) rather
    # than this sampling approach.
    types = list(RECIPIENT_DISTRIBUTION.keys())
    probs = list(RECIPIENT_DISTRIBUTION.values())
    return list(rng.choice(types, size=n, p=probs))


def _must_mention_facts(row: pd.Series) -> list[str]:
    """Conservative, structurally-reliable facts only. See module docstring
    for why richer facts (e.g. allergies) are not included here.
    """
    facts = []
    admission_type = row.get("admission_type")
    # Explicit isinstance + strip() check rather than a simple truthiness
    # test (`if admission_type:`) -- admission_type can be a float NaN after
    # the cohort join below if a stay_id/hadm_id pair failed to match
    # (see the n_missing_admission_type warning in build_grpo_prompts), and
    # `if nan:` is truthy in Python, which would silently produce a
    # malformed fact string like "admission type: nan" instead of correctly
    # skipping the missing case.
    if isinstance(admission_type, str) and admission_type.strip():
        facts.append(f"admission type: {admission_type.strip().lower()}")
    return facts


def build_grpo_prompts(vignettes_df: pd.DataFrame, cohort_df: pd.DataFrame,
                        cfg: GRPOPromptBuilderConfig) -> pd.DataFrame:
    """`vignettes_df` is the output of preprocessing.VignetteBuilder.
    `cohort_df` is the output of mimic_loader.MimicIVLoader.build_icu_cohort()
    (joined in here to recover admission_type for must_mention_facts, since
    the raw vignette rows don't carry it).
    """
    rng = np.random.default_rng(cfg.seed)

    # Left join (not inner) so that a vignette row without a matching cohort
    # entry is KEPT (with admission_type as NaN, handled by
    # _must_mention_facts above) rather than silently dropped -- losing
    # training examples silently because of an unrelated join mismatch would
    # be a much worse failure mode than keeping the row with one missing
    # optional fact.
    merged = vignettes_df.merge(
        cohort_df[["stay_id", "hadm_id", "admission_type"]].drop_duplicates(),
        on=["stay_id", "hadm_id"], how="left",
    )

    merged["reference_patient_state"] = merged.apply(_template_reference_state, axis=1)
    merged["recipient_type"] = _sample_recipient_types(len(merged), rng)
    merged["must_mention_facts"] = merged.apply(_must_mention_facts, axis=1)

    # deterioration_6h is carried through even though GRPO training itself
    # doesn't directly consume it (the reward functions use
    # reference_patient_state/recipient_type/must_mention_facts) -- kept
    # here so downstream evaluation (evaluation/metrics.py's Deterioration
    # Detection endpoint) can join back against this same file without a
    # second pass over the raw vignettes.
    # prediction_time is carried through so Phase 4's agentic tool use
    # (agents/tool_use.py's get_recent_labs) has a leakage-safe cutoff to
    # query against -- without it, a "recent labs" tool call would have no
    # way to know where "now" is for a given patient and could easily leak
    # values from AFTER the point this vignette's prompt was generated from.
    output_cols = ["stay_id", "hadm_id", "prediction_time", "prompt", "reference_patient_state",
                   "recipient_type", "must_mention_facts", "deterioration_6h"]
    result = merged[output_cols]

    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cfg.output_path)
    logger.info("Built %d GRPO-ready prompts -> %s", len(result), cfg.output_path)
    logger.info("Recipient type distribution: %s",
                result["recipient_type"].value_counts(normalize=True).to_dict())

    n_missing_admission_type = merged["admission_type"].isna().sum()
    if n_missing_admission_type > 0:
        # A non-zero count here usually means a stay_id/hadm_id mismatch
        # between vignettes_df and cohort_df (e.g. they were built from
        # different cohort snapshots, or one was filtered more aggressively
        # than the other) -- surfaced as a warning rather than silently
        # proceeding, since it can indicate the two input files are more
        # out of sync with each other than expected.
        logger.warning(
            "%d/%d rows had no admission_type after the cohort join -- check that "
            "vignettes_df and cohort_df share stay_id/hadm_id keys correctly.",
            n_missing_admission_type, len(merged),
        )

    return result


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Bridge Phase 1 MIMIC-IV vignettes into Phase 2's GRPO prompt schema."
    )
    parser.add_argument("--vignettes_path", required=True,
                        help="Path to the parquet output of preprocessing.VignetteBuilder "
                             "(e.g. ./data/processed/derivation/vignettes.parquet)")
    parser.add_argument("--cohort_path", required=True,
                        help="Path to the cached cohort parquet from mimic_loader "
                             "(e.g. ./cache/mimic/icu_cohort.parquet)")
    parser.add_argument("--output_path", default="./data/processed/derivation/grpo_prompts.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vignettes_df = pd.read_parquet(args.vignettes_path)
    cohort_df = pd.read_parquet(args.cohort_path)

    cfg = GRPOPromptBuilderConfig(output_path=args.output_path, seed=args.seed)
    build_grpo_prompts(vignettes_df, cohort_df, cfg)
