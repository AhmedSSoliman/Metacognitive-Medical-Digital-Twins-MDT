#!/bin/bash
#SBATCH --job-name=mdt-phase1-v2repro
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_v2repro_%j.out
#SBATCH --error=logs/phase1_v2repro_%j.err

set -euo pipefail
mkdir -p logs

module load conda
module load cudnn/9.6.0
conda activate /blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env

export LD_LIBRARY_PATH="/apps/compilers/cuda/12.8.1/lib64:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/cusparselt/lib:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export UNSLOTH_FORCE_FLOAT32=1

cd "$SLURM_SUBMIT_DIR"

# Reproduction test, not a new "version" -- as close as reconstructable to
# the ORIGINAL phase1_sft_v2 recipe that scored 97% format compliance:
#   - the original 36-vignette file (data/synthetic/stream_format_vignettes_original36_backup.jsonl),
#     NOT the 1000-vignette expansion used by v3/v4/diag_noanswer
#   - no --seed flag (v2 predates that argument's existence; HF's implicit
#     default of 42 was used)
#   - --no-soft_alignment: REQUIRED. First attempt at this reproduction
#     (job 39055683, 2026-08-10) omitted this flag on the wrong assumption
#     that v2 (which predates the flag's existence) matched the trainer's
#     CURRENT default -- but sft_trainer.py's --soft_alignment defaults to
#     True, so that run silently trained the already-known-broken two-stage
#     curriculum (the one that collapsed format compliance to 17.5%, see
#     the stage1 diagnostic in docs/development-log.md) on the 36-vignette
#     set instead of reproducing v2's true single-stage recipe. Caught via
#     the giveaway that a completed Stage 1 (`train_runtime`/`epoch: 2.0`
#     summary line) was immediately followed by the log restarting at
#     epoch 0 -- a Stage 2 that v2 never had. Killed and resubmitted with
#     this flag explicit. v2 itself, chronologically, only ever ran the
#     single-stage path (soft-alignment was added and tried AFTER v2), so
#     this flag is required to actually reproduce it, despite v2's own
#     sbatch script never having needed to pass it.
#
# Purpose: three full retrains (v3, v4, diag_noanswer) have all landed
# well below v2's reported 97%, and no single-variable test (seed change,
# answer-append flag at 1 epoch) has confirmed a specific cause. Before
# testing more variables, this checks the more basic question: does v2's
# ORIGINAL recipe still reproduce ~97% today, unchanged? If yes, the
# regression is confirmed to be caused specifically by something in the
# v3+ recipe changes (dataset expansion, most likely). If NO -- if this
# also lands well below 97% -- that points at environment/infrastructure
# drift (package versions, hardware/driver changes since 2026-07-19)
# affecting all recent runs equally, which would reframe the entire
# investigation. See docs/hypothesis_log.md.
python -u training/sft_trainer.py \
    --base_model google/medgemma-4b-it \
    --tier_one_dataset FreedomIntelligence/medical-o1-reasoning-SFT \
    --synthetic_vignettes ./data/synthetic/stream_format_vignettes_original36_backup.jsonl \
    --output_dir ./checkpoints/phase1_sft_v2_reproduction \
    --no-soft_alignment \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_r 32 \
    --lora_alpha 32 \
    --logging_steps 5 \
    --save_steps 15

echo "Phase 1 v2-reproduction job finished at $(date)"
