"""
Structural Empathy Reward (R_emp).

Evaluates readability calibration to user literacy level.
"""

import logging
import re
from typing import List

import torch

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

logger = logging.getLogger(__name__)


class StructuralEmpathyReward:
    """R_emp: Structural empathy reward based on readability."""
    
    def __init__(self):
        """Initialize structural empathy reward calculator."""
        # Target Flesch-Kincaid grade levels
        self.literacy_targets = {
            'low': (6.0, 8.0),
            'medium': (9.0, 12.0),
            'high': (13.0, 16.0)
        }
        logger.info("Initialized StructuralEmpathyReward")
    
    def compute(
        self,
        generated_texts: List[str],
        inferred_literacy: List[str],
        inferred_emotions: List[str]
    ) -> torch.Tensor:
        """Compute structural empathy rewards."""
        assert len(generated_texts) == len(inferred_literacy) == len(inferred_emotions)
        
        rewards = []
        
        for text, literacy, emotion in zip(generated_texts, inferred_literacy, inferred_emotions):
            # Readability match
            readability_reward = self._compute_readability_match(text, literacy)
            
            # Jargon penalty
            jargon_penalty = self._compute_jargon_penalty(text, literacy)
            
            # Combine
            reward = readability_reward - 0.2 * jargon_penalty
            rewards.append(max(0.0, reward))
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _compute_readability_match(self, text: str, literacy_level: str) -> float:
        """Compute readability match to target literacy."""
        if TEXTSTAT_AVAILABLE:
            try:
                fk_grade = textstat.flesch_kincaid_grade(text)
            except:
                return 0.5
        else:
            # Fallback
            words = text.split()
            if not words:
                return 0.5
            avg_word_len = sum(len(w) for w in words) / len(words)
            fk_grade = avg_word_len * 2.0
        
        # Get target range
        target_min, target_max = self.literacy_targets.get(literacy_level, (9.0, 12.0))
        
        # Compute match score
        if target_min <= fk_grade <= target_max:
            score = 1.0
        elif fk_grade < target_min:
            distance = target_min - fk_grade
            score = max(0.0, 1.0 - (distance / 5.0))
        else:
            distance = fk_grade - target_max
            score = max(0.0, 1.0 - (distance / 5.0))
        
        return score
    
    def _compute_jargon_penalty(self, text: str, literacy_level: str) -> float:
        """Penalize medical jargon for low literacy users."""
        jargon_patterns = [
            r'\b(myocardial|infarction|tachycardia|bradycardia)\b',
            r'\b(hypertension|hypotension|arrhythmia)\b',
            r'\b(nephropathy|hepatic|renal|cardiac)\b',
            r'\b(etiology|pathophysiology|pharmacology)\b',
            r'\b(hemodynamic|vasopressor|septicemia)\b'
        ]
        
        jargon_count = 0
        for pattern in jargon_patterns:
            jargon_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        word_count = len(text.split())
        if word_count == 0:
            return 0.0
        
        jargon_density = jargon_count / word_count
        
        if literacy_level == 'low':
            penalty = min(jargon_density * 5.0, 1.0)
        elif literacy_level == 'medium':
            penalty = min(jargon_density * 2.0, 1.0)
        else:
            penalty = 0.0
        
        return penalty