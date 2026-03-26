"""
Create GRPO training dataloader.

This script creates a dataloader with clinical prompts for GRPO training.
"""

import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class GRPOPromptDataset(Dataset):
    """Dataset of clinical prompts for GRPO training."""
    
    def __init__(self, prompts_file: str):
        """Load prompts from file."""
        with open(prompts_file, 'r') as f:
            self.prompts = json.load(f)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            'prompt': self.prompts[idx]['prompt'],
            'reference': self.prompts[idx].get('reference', '')
        }


def create_grpo_dataloader(
    prompts_file: str = "./outputs/mimic_processed.json",
    batch_size: int = 32,
    shuffle: bool = True
):
    """
    Create GRPO training dataloader.
    
    Args:
        prompts_file: Path to processed MIMIC prompts
        batch_size: Batch size for training
        shuffle: Whether to shuffle prompts
    
    Returns:
        DataLoader for GRPO training
    """
    dataset = GRPOPromptDataset(prompts_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Create dataloader
    dataloader = create_grpo_dataloader()
    print(f"✓ Created GRPO dataloader with {len(dataloader.dataset)} prompts")
    
    # Test
    batch = next(iter(dataloader))
    print(f"✓ Batch size: {len(batch['prompt'])}")
    print(f"✓ Sample prompt: {batch['prompt'][0][:100]}...")