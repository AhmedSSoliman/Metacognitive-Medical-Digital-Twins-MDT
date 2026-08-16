#!/bin/bash
#SBATCH --job-name=mdt-phase2-grpo
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --time=72:00:00
#SBATCH --output=logs/phase2_grpo_%j.out
#SBATCH --error=logs/phase2_grpo_%j.err

set -euo pipefail
mkdir -p logs

module load conda
module load cudnn/9.6.0
conda activate /blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env

export LD_LIBRARY_PATH="/apps/compilers/cuda/12.8.1/lib64:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/cusparselt/lib:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"

# See ../Agentic-DT_V1-July/slurm/phase2_grpo.sbatch for the full history of
# why this flag matters for Phase 2 (a real NaN collapse at step 12 without
# it, job 37933496) -- preserved verbatim there, not duplicated here.
export UNSLOTH_FORCE_FLOAT32=1

cd "$SLURM_SUBMIT_DIR"

# PORTED (2026-08-12) from ../Agentic-DT_V1-July/slurm/phase2_grpo.sbatch
# -- --base_model points at phase1_sft_v2_reproduction (the current working
# Phase 1 baseline; see ../Agentic-DT_V1-July/docs/hypothesis_log.md).
# --output_dir is a NEW directory (phase2_grpo_v2repro_base), not the
# original phase2_grpo, so this run can never clobber the earlier,
# already-evaluated GRPO checkpoints (checkpoint-130, final-200) that the
# research paper's Table 6 reports on.
#
# --kl_beta 0.01 (added 2026-08-12): previously training/grpo.py never set
# this, silently inheriting TRL's own beta=0.0 default (no KL penalty vs.
# the reference policy at all). Given this project's repeated real GRPO NaN
# collapses, an unconstrained policy is a plausible contributing factor --
# see training/grpo.py's --kl_beta argparse help and
# ../2026-04-24_nemotron-v1/ for where this specific value came from. Not
# validated on this backbone/recipe yet; if this run also collapses, that
# doesn't confirm beta=0.01 was wrong, but it's now at least a controlled,
# documented choice instead of a silent default.
#
# RESUME (2026-08-14): job 39279689 was CANCELLED (SIGNAL Terminated, not a
# crash) at step 83/200, well inside the 72h time limit -- clean
# checkpoint-80 exists (save_steps=5). --base_model here MUST stay
# ./checkpoints/phase1_sft_v2_reproduction_seed42_archived (the seed=42
# weights checkpoint-80 was actually trained from) even though
# ./checkpoints/phase1_sft_v2_reproduction now points at the seed=101
# baseline (92% format compliance, promoted 2026-08-14) -- resuming
# checkpoint-80's LoRA adapter against different base weights would silently
# corrupt the reference policy. RESUME_CHECKPOINT is unset by default so a
# fresh `sbatch scripts/run_grpo.sh` (e.g. a NEW run against the new seed=101
# baseline) still works with no flag; set it to resume this specific run:
#   sbatch --export=RESUME_CHECKPOINT=./checkpoints/phase2_grpo_v2repro_base/checkpoint-80 scripts/run_grpo.sh
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
BASE_MODEL="./checkpoints/phase1_sft_v2_reproduction"
# OUTPUT_DIR (added 2026-08-14): defaults to the resume run's directory so
# `sbatch --export=RESUME_CHECKPOINT=... scripts/run_grpo.sh` needs no other
# flags. A FRESH run (no RESUME_CHECKPOINT) against the seed=101 baseline
# MUST override this to a different directory -- both runs share the same
# default and would otherwise silently interleave checkpoint writes into
# the same folder:
#   sbatch --export=OUTPUT_DIR=./checkpoints/phase2_grpo_seed101_base scripts/run_grpo.sh
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/phase2_grpo_v2repro_base}"
RESUME_ARGS=()
if [ -n "$RESUME_CHECKPOINT" ]; then
    BASE_MODEL="./checkpoints/phase1_sft_v2_reproduction_seed42_archived"
    RESUME_ARGS=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

python -m data.build_grpo_prompts \
    --vignettes_path ./data/processed/derivation/vignettes.parquet \
    --cohort_path ./cache/mimic/icu_cohort.parquet \
    --output_path ./data/processed/derivation/grpo_prompts.parquet

python -m training.grpo \
    --base_model "$BASE_MODEL" \
    --prompt_dataset ./data/processed/derivation/grpo_prompts.parquet \
    --output_dir "$OUTPUT_DIR" \
    --num_generations 4 \
    --max_steps 200 \
    --save_steps 5 \
    --hypergraph_mode interim \
    --kl_beta 0.01 \
    "${RESUME_ARGS[@]}"

echo "Phase 2 GRPO job finished at $(date)"
