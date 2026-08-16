#!/bin/bash
#SBATCH --job-name=mdt-phase1v2repro-eval
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=16:00:00
#SBATCH --output=logs/phase1v2repro_eval_%j.out
#SBATCH --error=logs/phase1v2repro_eval_%j.err

set -euo pipefail
mkdir -p logs

module load conda
module load cudnn/9.6.0
conda activate /blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env
export LD_LIBRARY_PATH="/apps/compilers/cuda/12.8.1/lib64:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/cusparselt/lib:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"

# Real, statistically meaningful measurement (n=300) of the v2-reproduction
# checkpoint (job 39094537 -- v2's exact original single-stage recipe,
# original 36-vignette dataset, no seed, --no-soft_alignment explicitly
# set). Investigation into why v2's own recipe now scores 66.7% (n=30
# smoke eval) instead of its originally-reported 97% is documented in
# docs/hypothesis_log.md: recipe/dataset changes, seed variance, package
# versions, and base model weights have all been checked and ruled out as
# the cause; the eval/parsing code is a weak, unconfirmed lead. Given the
# GPU-hours already spent without resolving the discrepancy, this
# checkpoint (the best available, closest to the original validated
# recipe, clean 3-epoch run with no collapse) is being adopted as the new
# working Phase 1 baseline rather than pursuing further retrains. This
# n=300 run gets a precise, reportable number to actually use going
# forward (the n=30 smoke eval showed 66.7% format compliance, 0.716
# deterioration AUROC -- notably higher AUROC than v3's n=300 0.525,
# though at much smaller/noisier sample size).
python -u scripts/run_evaluation.py \
    --checkpoint ./checkpoints/phase1_sft_v2_reproduction \
    --eval_prompt_dataset ./data/processed/evaluation/grpo_prompts.parquet \
    --hypergraph_mode interim \
    --output_report ./evaluation/reports/phase1_sft_v2_reproduction_evaluation.json \
    --max_examples 300

echo "Phase 1 v2-reproduction full evaluation (n=300) finished at $(date)"
