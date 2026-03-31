#!/bin/bash
#SBATCH --job-name=medical_digital_twin
#SBATCH --account=prismap-ai-core
#SBATCH --qos=prismap-ai-core
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1  # Request B200 GPU (confirm availability)
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=job_output_%j.log
#SBATCH --error=job_error_%j.log

set -euo pipefail

# Ensure we are in submission directory
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "[$(date)] Starting job ${SLURM_JOB_ID:-N/A} on $(hostname)"

# Load and activate conda environment (robust for non-interactive shells)
module purge
module load conda

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda command not found after module load." >&2
    exit 1
fi

conda activate digitaltwins_env

echo "Python: $(which python)"
echo "Jupyter: $(which jupyter || true)"

# Avoid ~/.local site-packages shadowing env packages on compute nodes
export PYTHONNOUSERSITE=1

if ! command -v jupyter >/dev/null 2>&1; then
    echo "ERROR: jupyter is not available in digitaltwins_env." >&2
    exit 1
fi

# Run the notebook (assuming jupyter is available in the environment)
jupyter nbconvert \
    --to notebook \
    --execute medical_digital_twin_master.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    --output executed_notebook.ipynb

echo "[$(date)] Notebook execution completed. Output: executed_notebook.ipynb"

# Usage note: open in VS Code from login node after job finishes
echo "To open in VS Code from login node: code executed_notebook.ipynb"
