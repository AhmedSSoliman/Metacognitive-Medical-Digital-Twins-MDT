#!/bin/bash
#SBATCH --job-name=mdt-phase3-hypergraph
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-default    # CPU-only job -- hyperedge mining does not need a GPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=256gb                # combinatorial mining over chartevents-derived flags is memory-hungry
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase3_hypergraph_%j.out
#SBATCH --error=logs/phase3_hypergraph_%j.err

set -euo pipefail
mkdir -p logs

module load conda
conda activate /blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env

# Set MIMIC-IV dataset path default fallback
export MIMIC_ROOT="${MIMIC_ROOT:-/blue/prismap-ai-core/Ahmed/DigitalTwins/MDT/mimiciv/3.1}"

# COHORT_SCOPE controls whether hyperedge mining runs over the full ICU
# cohort ("full", the default) or only admissions with a cardiac-relevant
# diagnosis ("cardiac", per data/cohort_stratification.py's ICD
# categorization -- see that module and scripts/check_cohort_diagnostic_composition.py
# for why this distinction matters for the AHA cardiovascular framing).
export COHORT_SCOPE="${COHORT_SCOPE:-full}"
if [[ "$COHORT_SCOPE" != "full" && "$COHORT_SCOPE" != "cardiac" ]]; then
    echo "COHORT_SCOPE must be 'full' or 'cardiac', got '$COHORT_SCOPE'" >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"


# Step 1: build/cache the ICU cohort and temporal derivation/evaluation split
python data/mimic_loader.py \
    --mimic_root "$MIMIC_ROOT" \
    --cache_dir ./cache/mimic

# Step 2: extract vitals/labs time series for the DERIVATION partition only,
# after applying COHORT_SCOPE identically to both the derivation and
# evaluation partitions (see data/cohort_stratification.py's
# filter_by_category -- the same categorization Series is used for both, so
# neither partition gets special-cased treatment).
python -c "
from data.mimic_loader import MimicConfig, MimicIVLoader, temporal_split
from data.cohort_stratification import apply_cohort_scope
import os
import pandas as pd

mimic_root = os.environ['MIMIC_ROOT']
scope = os.environ['COHORT_SCOPE']

cfg = MimicConfig(mimic_root=mimic_root, cache_dir='./cache/mimic')
loader = MimicIVLoader(cfg)
cohort = loader.build_icu_cohort()
deriv, evalu = temporal_split(cohort)

deriv = apply_cohort_scope(mimic_root, deriv, scope)
evalu = apply_cohort_scope(mimic_root, evalu, scope)
print(f'COHORT_SCOPE={scope}: derivation n={len(deriv)}, evaluation n={len(evalu)}')

vitals = loader.extract_vitals_labs(deriv['stay_id'].tolist())
labs = loader.extract_labs_for_admissions(deriv['hadm_id'].tolist())
vitals.to_parquet('./cache/mimic/derivation_vitals.parquet')
labs.to_parquet('./cache/mimic/derivation_labs.parquet')
print('Derivation-partition time series extracted and cached.')

# vitals is stay_id-keyed, labs is hadm_id-keyed (see mimic_loader.py's
# extract_vitals_labs vs extract_labs_for_admissions) -- combine into one
# hadm_id-keyed frame via the cohort's stay_id<->hadm_id mapping, since
# hypergraph/construction.py's HyperedgeMiner.mine() takes a single id_col.
# Without this, mining only ever sees lab variables (lactate, creatinine,
# wbc, platelet, bilirubin, pH) and NEVER sees vitals (heart_rate, map,
# sbp, resp_rate, spo2, temperature) -- meaning this module's own flagship
# example hyperedge, {tachycardia, hypotension, hyperlactatemia}, could
# never actually be mined, since tachycardia/hypotension are vitals-only
# ABNORMALITY_RULES keys.
stay_to_hadm = deriv.set_index('stay_id')['hadm_id']
vitals_with_hadm = vitals.assign(hadm_id=vitals['stay_id'].map(stay_to_hadm)).dropna(subset=['hadm_id'])
vitals_with_hadm['hadm_id'] = vitals_with_hadm['hadm_id'].astype(int)
vitals_with_hadm = vitals_with_hadm[['hadm_id', 'charttime', 'variable', 'value']]
combined = pd.concat([vitals_with_hadm, labs[['hadm_id', 'charttime', 'variable', 'value']]], ignore_index=True)
combined.to_parquet('./cache/mimic/derivation_timeseries.parquet')
print(f'Combined vitals+labs timeseries: {len(combined)} rows ({len(vitals_with_hadm)} vitals + {len(labs)} labs) '
      f'-> ./cache/mimic/derivation_timeseries.parquet')
"

# Step 3: mine candidate hyperedges (status will be PENDING_CLINICAL_REVIEW --
# this does NOT auto-approve anything; see hypergraph/construction.py)
#
# --output_path is versioned per job (not the fixed derived_hypergraph.json)
# -- that fixed filename was found to have the SAME unversioned-overwrite
# risk that destroyed a validated Phase 1 checkpoint: IC3 clinical
# collaborators review derived_hypergraph.json in place, and *.json is
# globally gitignored (see .gitignore), so there was no VCS safety net if
# Phase 3 were ever re-run (different COHORT_SCOPE, refreshed MIMIC extract,
# etc) after a review pass. Promote a reviewed output to
# ./hypergraph/derived_hypergraph.json manually (or symlink it) once
# clinical review is complete -- never let this job write that path directly.
HYPERGRAPH_OUTPUT="./hypergraph/derived_hypergraph_${COHORT_SCOPE}_${SLURM_JOB_ID}.json"
python hypergraph/construction.py \
    --timeseries_path ./cache/mimic/derivation_timeseries.parquet \
    --output_path "$HYPERGRAPH_OUTPUT" \
    --min_support 30

echo "Phase 3 hyperedge mining finished at $(date) (COHORT_SCOPE=$COHORT_SCOPE). Hyperedges are PENDING_CLINICAL_REVIEW."
echo "Next step: have IC3 clinical collaborators review $HYPERGRAPH_OUTPUT"
echo "before promoting it to ./hypergraph/derived_hypergraph.json for use in Phase 2/4 training"
echo "(see hypergraph/construction.py FaceValidityReview). Do NOT overwrite an existing reviewed file."
