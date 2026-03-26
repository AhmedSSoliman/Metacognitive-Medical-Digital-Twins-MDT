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
    from bert_score import BERTScorer
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
            
        if BERT_SCORE_AVAILABLE:
            # Initialize BERTScorer once to heavily accelerate training and prevent repeating downloads
            # We use 'distilroberta-base' to speed up the reward function, or default to 'roberta-large'.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.bert_scorer = BERTScorer(
                    lang='en', 
                    rescale_with_baseline=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
        else:
            self.bert_scorer = None
        
        logger.info(f"Initialized SemanticFidelityReward")
    
    def compute(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> torch.Tensor:
        """Compute semantic fidelity rewards."""
        assert len(generated_texts) == len(reference_texts)
        
        # Batch pass for BERTScore to be overwhelmingly faster
        bert_f1_scores = [0.0] * len(generated_texts)
        if self.bert_scorer is not None and len(generated_texts) > 0:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    P, R, F1 = self.bert_scorer.score(generated_texts, reference_texts)
                    bert_f1_scores = F1.tolist()
            except Exception as e:
                logger.warning(f"BERTScore failed: {e}")
        
        rewards = []
        
        for i, (gen, ref) in enumerate(zip(generated_texts, reference_texts)):
            if self.rouge_scorer:
                rouge_scores = self.rouge_scorer.score(ref, gen)
                rouge_l_f1 = rouge_scores['rougeL'].fmeasure
            else:
                rouge_l_f1 = self._simple_overlap(gen, ref)
            
            if self.bert_scorer is not None:
                bert_f1 = bert_f1_scores[i]
                reward = 0.5 * rouge_l_f1 + 0.5 * bert_f1
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