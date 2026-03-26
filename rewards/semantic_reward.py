"""
Semantic Fidelity Reward (R_semantic).

Uses ROUGE-L and BERTScore to measure alignment with expert responses.
"""

import logging
from typing import List

import torch

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticFidelityReward:
    """R_semantic: Semantic fidelity reward using ROUGE-L and BERTScore."""
    
    def __init__(self, rouge_types: List[str] = None):
        """Initialize semantic fidelity reward calculator."""
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                self.rouge_types,
                use_stemmer=True
            )
        else:
            self.rouge_scorer = None
            logger.warning("ROUGE not available, using fallback")
        
        logger.info(f"Initialized SemanticFidelityReward")
    
    def compute(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> torch.Tensor:
        """Compute semantic fidelity rewards."""
        assert len(generated_texts) == len(reference_texts)
        
        rewards = []
        
        for gen, ref in zip(generated_texts, reference_texts):
            if self.rouge_scorer:
                rouge_scores = self.rouge_scorer.score(ref, gen)
                rouge_l_f1 = rouge_scores['rougeL'].fmeasure
            else:
                rouge_l_f1 = self._simple_overlap(gen, ref)
            
            if BERT_SCORE_AVAILABLE:
                try:
                    P, R, F1 = bert_score(
                        [gen], [ref],
                        lang='en',
                        verbose=False,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    bert_f1 = F1.item()
                    reward = 0.5 * rouge_l_f1 + 0.5 * bert_f1
                except Exception as e:
                    logger.warning(f"BERTScore failed: {e}")
                    reward = rouge_l_f1
            else:
                reward = rouge_l_f1
            
            rewards.append(reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def _simple_overlap(self, text1: str, text2: str) -> float:
        """Fallback simple word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0