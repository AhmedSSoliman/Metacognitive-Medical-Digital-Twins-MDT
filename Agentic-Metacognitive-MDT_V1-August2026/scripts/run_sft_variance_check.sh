#!/bin/bash
#SBATCH --job-name=mdt-phase1-variance
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --output=logs/phase1_variance_%j.out
#SBATCH --error=logs/phase1_variance_%j.err

set -euo pipefail
mkdir -p logs

module load conda
module load cudnn/9.6.0
conda activate /blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env

export LD_LIBRARY_PATH="/apps/compilers/cuda/12.8.1/lib64:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/cusparselt/lib:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"
export UNSLOTH_FORCE_FLOAT32=1

cd "$SLURM_SUBMIT_DIR"

# Run-to-run VARIANCE check (added 2026-08-13, per docs/hypothesis_log.md's
# remaining open question): every prior test of the v2/v3+ gap used a
# SINGLE run per configuration -- one data point cannot distinguish "this
# recipe reliably produces ~46.7%" from "this recipe is highly unstable
# and 46.7% vs the original 97% are both within its natural range." This
# runs the IDENTICAL recipe as scripts/run_sft.sh (same dataset, same
# --no-soft_alignment, same hyperparameters) with ONLY --seed varied, via
# the SEED_VALUE env var. Submit 2-3 times with different seeds:
#   sbatch --export=SEED_VALUE=101 scripts/run_sft_variance_check.sh
#   sbatch --export=SEED_VALUE=202 scripts/run_sft_variance_check.sh
#   sbatch --export=SEED_VALUE=303 scripts/run_sft_variance_check.sh
# If format compliance across these 3 runs clusters tightly (e.g. all
# 40-55%), that's strong evidence 46.7% IS this recipe's real, stable
# behavior today, and the original 97% was something else entirely
# (environment at the time, a different unrecorded recipe detail, or
# simply an outlier). If they spread widely (e.g. one lands near 90%),
# that reopens the whole investigation with a genuinely new data point.
SEED_VALUE="${SEED_VALUE:?Set SEED_VALUE, e.g. sbatch --export=SEED_VALUE=101 scripts/run_sft_variance_check.sh}"

python -m training.sft \
    --base_model google/medgemma-4b-it \
    --tier_one_dataset FreedomIntelligence/medical-o1-reasoning-SFT \
    --synthetic_vignettes ./data/synthetic/stream_format_vignettes_original36_backup.jsonl \
    --output_dir "./checkpoints/phase1_sft_variance_seed${SEED_VALUE}" \
    --no-soft_alignment \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lora_r 32 \
    --lora_alpha 32 \
    --logging_steps 5 \
    --save_steps 15 \
    --seed "$SEED_VALUE"

echo "Phase 1 variance-check run (seed=$SEED_VALUE) finished at $(date)"
