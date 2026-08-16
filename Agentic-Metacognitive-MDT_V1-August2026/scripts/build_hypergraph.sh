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

export MIMIC_ROOT="${MIMIC_ROOT:-/blue/prismap-ai-core/Ahmed/DigitalTwins/MDT/mimiciv/3.1}"

# COHORT_SCOPE: "full" (default) or "cardiac" -- see core/cohort/mimic.py's
# cohort categorization and scripts/check_cohort_diagnostic_composition.py.
export COHORT_SCOPE="${COHORT_SCOPE:-full}"
if [[ "$COHORT_SCOPE" != "full" && "$COHORT_SCOPE" != "cardiac" ]]; then
    echo "COHORT_SCOPE must be 'full' or 'cardiac', got '$COHORT_SCOPE'" >&2
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"

# PORTED (2026-08-12) from ../Agentic-DT_V1-July/slurm/phase3_hypergraph.sbatch
# -- module paths updated for the new layout: data.mimic_loader ->
# core.cohort.mimic, hypergraph.construction -> core.hypergraph.construction.
# See that source file (preserved in the source repo) for the full
# vitals+labs-combination rationale this step depends on -- omitting it
# means mining only ever sees lab variables and never vitals, so the
# flagship {tachycardia, hypotension, hyperlactatemia} hyperedge could
# never be mined at all.

# Step 1: build/cache the ICU cohort and temporal derivation/evaluation split
python -m core.cohort.mimic \
    --mimic_root "$MIMIC_ROOT" \
    --cache_dir ./cache/mimic

# Step 2: extract vitals/labs time series for the DERIVATION partition only,
# apply COHORT_SCOPE identically to both partitions, combine vitals (stay_id
# -keyed) and labs (hadm_id-keyed) into one hadm_id-keyed frame.
python -c "
from core.cohort.mimic import MimicConfig, MimicIVLoader, temporal_split
from core.cohort.terminology import apply_cohort_scope
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
# this does NOT auto-approve anything; see core/hypergraph/construction.py).
# --output_path is versioned per job -- never write the fixed
# derived_hypergraph.json path directly; promote a reviewed output to it
# manually once clinical review is complete.
HYPERGRAPH_OUTPUT="./hypergraph/derived_hypergraph_${COHORT_SCOPE}_${SLURM_JOB_ID}.json"
python -m core.hypergraph.construction \
    --timeseries_path ./cache/mimic/derivation_timeseries.parquet \
    --output_path "$HYPERGRAPH_OUTPUT" \
    --min_support 30

echo "Phase 3 hyperedge mining finished at $(date) (COHORT_SCOPE=$COHORT_SCOPE). Hyperedges are PENDING_CLINICAL_REVIEW."
echo "Next step: have IC3 clinical collaborators review $HYPERGRAPH_OUTPUT"
echo "before promoting it to ./hypergraph/derived_hypergraph.json for use in Phase 2/4 training."
echo "Do NOT overwrite an existing reviewed file."
