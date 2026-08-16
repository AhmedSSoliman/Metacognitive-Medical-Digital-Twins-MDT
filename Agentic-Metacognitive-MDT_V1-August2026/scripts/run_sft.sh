#!/bin/bash
#SBATCH --job-name=mdt-phase1-sft
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_sft_%j.out
#SBATCH --error=logs/phase1_sft_%j.err

set -euo pipefail
mkdir -p logs

module load conda
module load cudnn/9.6.0
conda activate /blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env

export LD_LIBRARY_PATH="/apps/compilers/cuda/12.8.1/lib64:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/cusparselt/lib:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export UNSLOTH_FORCE_FLOAT32=1

cd "$SLURM_SUBMIT_DIR"

# PORTED (2026-08-12) from ../Agentic-DT_V1-July/slurm/phase1_sft_v2_reproduction.sbatch
# -- this is the current working baseline recipe (phase1_sft_v2_reproduction,
# 46.7% format compliance at n=300, confirmed 2026-08-12; see
# ../Agentic-DT_V1-July/docs/hypothesis_log.md and PROGRESS.md for why this,
# not the originally-reported 97%, is the honest baseline). A run-to-run
# variance check (scripts/run_sft_variance_check.sh, seeds 101/202, jobs
# 39288830/39288831) is testing whether 46.7% is this recipe's stable
# behavior or itself just one noisy data point -- see PROGRESS.md for the
# result once available. --no-soft_alignment and the original 36-vignette
# dataset are REQUIRED to match this recipe -- omitting --no-soft_alignment
# silently runs a different, already-known-broken two-stage curriculum
# (caught once during this investigation; see the hypothesis log).
python -m training.sft \
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

echo "Phase 1 SFT job finished at $(date)"
