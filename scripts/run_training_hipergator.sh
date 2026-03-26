#!/bin/bash
#SBATCH --job-name=mdt-training
#SBATCH --account=your-group        # Update this
#SBATCH --qos=your-group-b         # Update this
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/mdt_%j.out
#SBATCH --error=logs/mdt_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@ufl.edu  # Update this

# Load modules
module load conda
module load cuda/11.8

# Activate environment
conda activate mdt-env

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
## cd /blue/your-group/your-username/medical-digital-twin
cd /blue/prismap-ai-core/ahmed.soliman/Ahmed/DigitalTwins/MDT


# Create log directory
mkdir -p logs

# Run training
echo "Starting SFT training at $(date)"
python main.py --train-sft \
    --max-patients 1000 \
    --max-o1-examples 5000 \
    2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log

echo "Training completed at $(date)"