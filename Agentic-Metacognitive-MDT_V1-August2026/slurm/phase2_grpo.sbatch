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

# Explicitly link CUDA 12.8.1 and local environment nvidia libraries
export LD_LIBRARY_PATH="/apps/compilers/cuda/12.8.1/lib64:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/cusparselt/lib:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"

# UNSLOTH_FORCE_FLOAT32 -- history of this flag for Phase 2, in order:
# 1. Initially NOT set, betting that Phase 2 (starting from an already-
#    converged checkpoint) wouldn't need it the way Phase 1's from-scratch
#    SFT did. Job 37933496 proved this wrong: 11 genuinely improving steps
#    (format=1.0, semantic/physio rewards trending up, grad_norm stable
#    0.24-0.43), then NaN collapse at step 12 -- the same ~10-20-step
#    collapse signature Phase 1 hit without this flag.
# 2. Re-enabling it was first blocked by a real bug in THIS project's own
#    code (training/grpo_trainer.py re-read the env var after model
#    loading, but unsloth/models/loader.py's FastLanguageModel.from_pretrained()
#    unconditionally resets it internally -- fixed by capturing it into a
#    frozen module-level constant before any Unsloth import/load happens;
#    see that file's matching comment).
# 3. With the real fix in place (job 38036930), re-testing on this exact
#    B200 + corrected-prompt setup: 7+ clean steps past the step-12 danger
#    zone, grad_norm healthy (0.28-0.60), semantic_reward climbing 0.26 ->
#    0.60 -- genuinely BETTER training signal than the unstable run. Cost:
#    ~1200s/step (~12x slower than without it, since it also requires
#    disabling KV caching to avoid a matching Cache.update() dtype crash --
#    see training/grpo_trainer.py's generation_kwargs logic), meaning a full
#    200-step run takes ~67h, close to this job's 72h limit. Accepted given
#    the training signal quality; save_steps is set low below specifically
#    to keep any walltime-driven interruption cheap to resume from (just
#    point --base_model at the latest checkpoints/phase2_grpo/checkpoint-N).
export UNSLOTH_FORCE_FLOAT32=1


cd "$SLURM_SUBMIT_DIR"


# Step 1: bridge Phase 1's raw MIMIC-IV vignettes into the schema Phase 2's
# GRPO trainer requires (reference_patient_state, recipient_type,
# must_mention_facts -- none of which exist in Phase 1's raw output).
# See data/build_grpo_prompts.py for what is templated vs. clinically reviewed.
python data/build_grpo_prompts.py \
    --vignettes_path ./data/processed/derivation/vignettes.parquet \
    --cohort_path ./cache/mimic/icu_cohort.parquet \
    --output_path ./data/processed/derivation/grpo_prompts.parquet

# Step 2: run GRPO training directly against Phase 1's LoRA adapter
# checkpoint -- NOT a merged model. scripts/merge_phase1_checkpoint.py's
# merge step (both Unsloth's save_pretrained_merged AND PEFT's
# merge_and_unload) was confirmed via scripts/validate_phase1_checkpoint.py
# to silently corrupt the model (0/4 format-compliant, degenerate output),
# while loading this SAME adapter directly is 4/4 compliant with genuinely
# good clinical content -- see training/grpo_trainer.py's matching comment
# in load_model_and_tokenizer_unsloth(). Phase 2 now continues training
# Phase 1's existing adapter (no merge, no second fresh LoRA) instead.
#
# --base_model updated to phase1_sft_v2_reproduction (2026-08-12): the
# original ./checkpoints/phase1_sft was stale and pre-dated the whole
# v2/v3/v4 investigation. That investigation (see
# docs/hypothesis_log.md) found that v3/v4/diag_noanswer all regressed
# well below v2's originally-reported 97% format compliance, and -- after
# ruling out recipe/dataset changes, seed variance, package version drift,
# and base model weight drift as causes -- v2's OWN unchanged recipe,
# retrained today, also only reaches ~66.7% (n=30 smoke eval;
# checkpoints/phase1_sft_v2_reproduction). The root cause of the gap vs.
# the original 97% was not found despite exhausting every cheap
# (no-GPU-hours) hypothesis. Given GPU-hours already spent on four full
# retrains, phase1_sft_v2_reproduction (closest to the original validated
# recipe, clean 3-epoch run, GradientCollapseDetector never triggered) is
# being adopted as the working baseline rather than pursuing further
# retrains -- see docs/hypothesis_log.md for the full trail.
python training/grpo_trainer.py \
    --base_model ./checkpoints/phase1_sft_v2_reproduction \
    --prompt_dataset ./data/processed/derivation/grpo_prompts.parquet \
    --output_dir ./checkpoints/phase2_grpo_v2repro_base \
    --num_generations 4 \
    --max_steps 200 \
    --save_steps 5 \
    --hypergraph_mode interim

echo "Phase 2 GRPO job finished at $(date)"
