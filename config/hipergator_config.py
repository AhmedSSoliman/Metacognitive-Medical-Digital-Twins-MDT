"""
HiPerGator-specific configuration for MIMIC-IV processing.

Update the paths to match your HiPerGator directory structure.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HiPerGatorConfig:
    """HiPerGator environment configuration."""
    
    # Update these paths for your setup
    user_group: str = "prismap-ai-core"  # e.g., "your-group"
    username: str = "ahmed.soliman"  # e.g., "your-username"
    
    # MIMIC-IV location on HiPerGator
    mimic_root: str = f"/blue/{user_group}/{username}/Ahmed/DigitalTwins/MDT/mimic-iv-3.1"
    
    # Scratch space for temporary files (fast I/O)
    scratch_dir: str = f"/blue/{user_group}/{username}/Ahmed/DigitalTwins/MDT/scratch"
    
    # Output directory for processed data
    output_dir: str = f"/blue/{user_group}/{username}/Ahmed/DigitalTwins/MDT/mdt-outputs"
    
    # GPU partition
    gpu_partition: str = "gpu"  # or "hpg-ai" for A100s
    
    # Resource allocation
    num_gpus: int = 1
    gpu_type: str = "a100"  # or "rtx6000"
    memory_gb: int = 64
    time_hours: int = 24
    
    def get_mimic_config(self):
        """Get DataConfig with HiPerGator paths."""
        from config.configs import DataConfig
        
        return DataConfig(
            mimic_root_dir=self.mimic_root,
            mimic_version="3.1"
        )