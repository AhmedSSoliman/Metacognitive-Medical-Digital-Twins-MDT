"""
Proactivity Reward (R_physio).

Rewards early prediction of physiological deterioration.
"""

import logging
from typing import Dict

import torch

logger = logging.getLogger(__name__)


class ProactivityReward:
    """R_physio: Proactive surge prediction reward."""
    
    def __init__(
        self,
        surge_thresholds: Dict[str, float] = None,
        anticipation_window_hours: int = 4
    ):
        """Initialize proactivity reward calculator."""
        self.surge_thresholds = surge_thresholds or {
            'lactate': 2.0,
            'creatinine': 1.5,
            'heart_rate': 120,
            'sbp': 90,
            'spo2': 90
        }
        self.anticipation_window = anticipation_window_hours
        logger.info(f"Initialized ProactivityReward")
    
    def compute(
        self,
        predictions: Dict[str, torch.Tensor],
        ground_truth: Dict[str, torch.Tensor],
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Compute proactivity rewards."""
        batch_size = next(iter(predictions.values())).shape[0]
        rewards = torch.zeros(batch_size, dtype=torch.float32)
        
        for biomarker, threshold in self.surge_thresholds.items():
            if biomarker not in predictions or biomarker not in ground_truth:
                continue
            
            pred = predictions[biomarker]
            truth = ground_truth[biomarker]
            
            pred_surges = self._detect_surges(pred, threshold, biomarker)
            truth_surges = self._detect_surges(truth, threshold, biomarker)
            
            biomarker_rewards = self._compute_anticipation_rewards(
                pred_surges,
                truth_surges,
                timestamps
            )
            
            rewards += biomarker_rewards
        
        # Normalize
        rewards = rewards / len(self.surge_thresholds)
        
        return rewards
    
    def _detect_surges(
        self,
        values: torch.Tensor,
        threshold: float,
        biomarker: str
    ) -> torch.Tensor:
        """Detect surge events."""
        if biomarker in ['sbp', 'spo2']:
            surges = values < threshold
        else:
            surges = values > threshold
        
        return surges
    
    def _compute_anticipation_rewards(
        self,
        pred_surges: torch.Tensor,
        truth_surges: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Reward early prediction."""
        batch_size = pred_surges.shape[0]
        rewards = torch.zeros(batch_size, dtype=torch.float32)
        
        for i in range(batch_size):
            pred = pred_surges[i]
            truth = truth_surges[i]
            
            pred_times = torch.where(pred)[0]
            if len(pred_times) == 0:
                continue
            
            first_pred = pred_times[0].item()
            
            truth_times = torch.where(truth)[0]
            if len(truth_times) == 0:
                continue
            
            first_truth = truth_times[0].item()
            
            anticipation = first_truth - first_pred
            
            if 0 < anticipation <= self.anticipation_window:
                rewards[i] = anticipation / self.anticipation_window
            elif anticipation > self.anticipation_window:
                rewards[i] = 0.5
        
        return rewards