#!/bin/bash
#SBATCH --job-name=medical_digital_twin
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1  # Request B200 GPU (confirm availability)
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=job_output_%j.log
#SBATCH --error=job_error_%j.log

# Load environment
module load conda
conda activate digitaltwins_env

# Run the notebook (assuming jupyter is available in the environment)
jupyter nbconvert --to notebook --execute medical_digital_twin_master.ipynb --output executed_notebook.ipynb
