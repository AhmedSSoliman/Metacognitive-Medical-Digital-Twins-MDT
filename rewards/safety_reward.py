"""
Biological Safety Reward (R_bound).

Enforces physiological plausibility and prevents hallucinations.
"""

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


class BiologicalSafetyReward:
    """R_bound: Biological safety constraint reward."""
    
    def __init__(self, reference_ranges: Dict[str, Tuple[float, float]] = None):
        """Initialize biological safety reward calculator."""
        self.reference_ranges = reference_ranges or {
            'heart_rate': (30, 250),
            'sbp': (50, 250),
            'dbp': (20, 150),
            'spo2': (50, 100),
            'lactate': (0.0, 20.0),
            'creatinine': (0.0, 15.0),
            'temperature': (32.0, 42.0),
            'respiratory_rate': (5, 60)
        }
        
        self.incompatible_combinations = [
            ('heart_rate', 'sbp', self._check_brady_hypotension),
            ('spo2', 'respiratory_rate', self._check_hypoxia_bradypnea)
        ]
        
        logger.info("Initialized BiologicalSafetyReward")
    
    def compute(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute biological safety scores."""
        batch_size = next(iter(predictions.values())).shape[0]
        safety_scores = torch.ones(batch_size, dtype=torch.float32)
        
        # Check individual ranges
        for biomarker, values in predictions.items():
            if biomarker not in self.reference_ranges:
                continue
            
            min_val, max_val = self.reference_ranges[biomarker]
            violations = (values < min_val) | (values > max_val)
            
            penalties = self._compute_violation_penalties(
                values,
                min_val,
                max_val,
                violations
            )
            
            safety_scores *= (1.0 - penalties)
        
        # Check combinations
        for marker1, marker2, check_func in self.incompatible_combinations:
            if marker1 in predictions and marker2 in predictions:
                combination_penalties = check_func(
                    predictions[marker1],
                    predictions[marker2]
                )
                safety_scores *= (1.0 - combination_penalties)
        
        safety_scores = torch.clamp(safety_scores, 0.0, 1.0)
        
        return safety_scores
    
    def _compute_violation_penalties(
        self,
        values: torch.Tensor,
        min_val: float,
        max_val: float,
        violations: torch.Tensor
    ) -> torch.Tensor:
        """Compute penalties for range violations."""
        penalties = torch.zeros_like(values, dtype=torch.float32)
        
        below_mask = values < min_val
        if below_mask.any():
            distance = (min_val - values[below_mask]) / min_val
            penalties[below_mask] = torch.clamp(distance, 0.0, 1.0)
        
        above_mask = values > max_val
        if above_mask.any():
            distance = (values[above_mask] - max_val) / max_val
            penalties[above_mask] = torch.clamp(distance, 0.0, 1.0)
        
        return penalties
    
    def _check_brady_hypotension(
        self,
        heart_rate: torch.Tensor,
        sbp: torch.Tensor
    ) -> torch.Tensor:
        """Check for impossible bradycardia + hypotension."""
        brady_hypotension = (heart_rate < 40) & (sbp < 80)
        
        penalties = torch.where(
            brady_hypotension,
            torch.ones_like(heart_rate) * 0.8,
            torch.zeros_like(heart_rate)
        )
        
        return penalties
    
    def _check_hypoxia_bradypnea(
        self,
        spo2: torch.Tensor,
        respiratory_rate: torch.Tensor
    ) -> torch.Tensor:
        """Check for unlikely hypoxia + bradypnea."""
        hypoxia_bradypnea = (spo2 < 85) & (respiratory_rate < 10)
        
        penalties = torch.where(
            hypoxia_bradypnea,
            torch.ones_like(spo2) * 0.7,
            torch.zeros_like(spo2)
        )
        
        return penalties
    
    def detect_hallucinations(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> Dict[str, List[int]]:
        """Detect and categorize hallucinations."""
        hallucinations = {
            'range_violations': [],
            'impossible_combinations': [],
            'extreme_values': []
        }
        
        batch_size = next(iter(predictions.values())).shape[0]
        
        for i in range(batch_size):
            has_violation = False
            has_combination = False
            has_extreme = False
            
            for biomarker, values in predictions.items():
                if biomarker not in self.reference_ranges:
                    continue
                
                value = values[i].item()
                min_val, max_val = self.reference_ranges[biomarker]
                
                if value < min_val or value > max_val:
                    has_violation = True
                
                range_width = max_val - min_val
                if value < (min_val - range_width) or value > (max_val + range_width):
                    has_extreme = True
            
            for marker1, marker2, check_func in self.incompatible_combinations:
                if marker1 in predictions and marker2 in predictions:
                    penalty = check_func(
                        predictions[marker1][i:i+1],
                        predictions[marker2][i:i+1]
                    )
                    if penalty.item() > 0.5:
                        has_combination = True
            
            if has_violation:
                hallucinations['range_violations'].append(i)
            if has_combination:
                hallucinations['impossible_combinations'].append(i)
            if has_extreme:
                hallucinations['extreme_values'].append(i)
        
        return hallucinations