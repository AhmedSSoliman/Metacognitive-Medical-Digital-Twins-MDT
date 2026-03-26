# HiPerGator Setup Guide

## 1. Connect to HiPerGator
```bash
ssh your-username@hpg.rc.ufl.edu
```

## 2. Navigate to Blue Storage
```bash
cd /blue/your-group/your-username
```

## 3. Clone Repository
```bash
git clone https://github.com/ahmedsoliman/medical-digital-twin.git
cd medical-digital-twin
```

## 4. Create Conda Environment
```bash
module load conda
conda create -n mdt-env python=3.10
conda activate mdt-env
```

## 5. Install Dependencies
```bash
pip install -r requirements.txt
```

## 6. Update Configuration

Edit `config/hipergator_config.py`:
```python
user_group: str = "ufhobictr"  # Your group
username: str = "asoliman"      # Your username
```

## 7. Verify MIMIC-IV Access
```bash
python -c "
from data.mimic_processor import MIMICProcessor
from config.hipergator_config import HiPerGatorConfig

config = HiPerGatorConfig().get_mimic_config()
processor = MIMICProcessor(config)
print('MIMIC-IV available:', processor.check_availability())
"
```

## 8. Submit Training Job
```bash
# Update paths in the script
nano scripts/run_training_hipergator.sh

# Create logs directory
mkdir -p logs

# Submit job
sbatch scripts/run_training_hipergator.sh

# Check status
squeue -u $USER

# View output
tail -f logs/mdt_<jobid>.out
```

## 9. Interactive Session (for testing)
```bash
srun --partition=gpu --gpus=a100:1 --mem=32gb --time=04:00:00 --pty bash -i

# Activate environment
module load conda
conda activate mdt-env

# Run tests
python main.py --test-only
```