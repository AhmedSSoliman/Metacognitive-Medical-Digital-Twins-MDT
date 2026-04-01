"""
PyTorch Dataset for cognitive stream training.
"""

import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CognitiveStreamDataset(Dataset):
    """PyTorch Dataset for cognitive stream training."""
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_length: int = 2048
    ):
        """Initialize dataset."""
        self.data = data
        self.tokenizer = tokenizer
        # Cap requested length to tokenizer limit to avoid position id overflow (e.g., GPT-2 has 1024)
        tokenizer_limit = getattr(getattr(tokenizer, "model_max_length", None), "__int__", lambda: None)()
        if tokenizer_limit is None:
            tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if tokenizer_limit is None:
            tokenizer_limit = max_length
        self.max_length = min(max_length, int(tokenizer_limit))
        
        logger.info(f"Initialized dataset with {len(data)} examples")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training example."""
        example = self.data[idx]
        
        # Construct prompt
        prompt = self._construct_prompt(example)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels for causal LM
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'prompt': prompt,  # Include raw prompt for GRPO training
            'think_weight': torch.tensor(float(example.get('think_weight', 1.0)), dtype=torch.float32)
        }
    
    def _construct_prompt(self, example: Dict) -> str:
        """Construct prompt with cognitive streams."""
        prompt = f"Clinical Case:\n{example.get('case_description', '')}\n\n"

        reasoning_text = (
            example.get('reasoning')
            or example.get('think')
            or example.get('think_synth')
            or ''
        )

        prompt += f"<think>\n{reasoning_text}\n</think>\n\n"
        prompt += f"<patient_state>\n{example.get('patient_state', '')}\n</patient_state>\n\n"
        prompt += f"<user_belief>\n{example.get('user_belief', '')}\n</user_belief>\n\n"
        
        if 'response' in example:
            prompt += f"Response:\n{example['response']}"
        
        return prompt