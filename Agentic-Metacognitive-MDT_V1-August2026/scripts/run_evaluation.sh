#!/bin/bash
#SBATCH --job-name=mdt-phase1-eval
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=16:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail
mkdir -p logs

module load conda
module load cudnn/9.6.0
conda activate /blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env
export LD_LIBRARY_PATH="/apps/compilers/cuda/12.8.1/lib64:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/cusparselt/lib:/blue/prismap-ai-core/Ahmed/envs/prismap_digitaltwin_env/lib/python3.13/site-packages/nvidia/nvshmem/lib:${LD_LIBRARY_PATH:-}"

cd "$SLURM_SUBMIT_DIR"

# PORTED (2026-08-12) from
# ../Agentic-DT_V1-July/slurm/run_phase1_v2_reproduction_evaluation.sbatch
# -- real n=300 measurement against the current working baseline checkpoint.
# Override CHECKPOINT/OUTPUT_REPORT env vars to evaluate a different one:
#   sbatch --export=CHECKPOINT=./checkpoints/other,OUTPUT_REPORT=./evaluation/reports/other.json scripts/run_evaluation.sh
CHECKPOINT="${CHECKPOINT:-./checkpoints/phase1_sft_v2_reproduction}"
OUTPUT_REPORT="${OUTPUT_REPORT:-./evaluation/reports/phase1_sft_v2_reproduction_evaluation.json}"

python -u evaluation/run_evaluation.py \
    --checkpoint "$CHECKPOINT" \
    --eval_prompt_dataset ./data/processed/evaluation/grpo_prompts.parquet \
    --hypergraph_mode interim \
    --output_report "$OUTPUT_REPORT" \
    --max_examples 300

echo "Evaluation (n=300) of $CHECKPOINT finished at $(date), report at $OUTPUT_REPORT"
