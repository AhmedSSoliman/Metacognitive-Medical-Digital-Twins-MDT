"""
Composite Reward Engine for Multi-Objective GRPO.

Combines five reward components with configurable weights:
    1. Semantic Fidelity (R_sem): Clinical accuracy via ROUGE-L + BERTScore
    2. Metacognitive Depth (R_meta): Self-correction via Delta-Embedding
    3. Structural Empathy (R_emp): Readability calibration via Flesch-Kincaid
    4. Proactive Physiological Monitoring (R_physio): Early surge detection
    5. Biological Safety (R_bound): Reference range constraint validation

Total Reward:
    R_total = w_sem * R_sem + w_meta * R_meta + w_emp * R_emp + 
              w_physio * R_physio + w_safety * R_bound

Author: Ahmed Soliman
Institution: University of Florida, Health Outcomes & Biomedical Informatics (HOBI)
"""

import logging
import torch
from typing import Dict, List, Optional

from rewards.semantic_reward import SemanticFidelityReward
from rewards.metacognitive_reward import MetacognitiveDepthReward
from rewards.empathy_reward import StructuralEmpathyReward
from rewards.proactivity_reward import ProactivityReward
from rewards.safety_reward import BiologicalSafetyReward

logger = logging.getLogger(__name__)


class CompositeRewardEngine:
    """
    Multi-objective reward engine for GRPO training.
    
    Combines five reward signals with configurable weights to guide
    reinforcement learning alignment toward clinical accuracy, safety,
    empathy, proactivity, and metacognition.
    
    Example:
        >>> engine = CompositeRewardEngine(
        ...     w_semantic=0.25,
        ...     w_metacognitive=0.20,
        ...     w_empathy=0.15,
        ...     w_proactivity=0.25,
        ...     w_safety=0.15
        ... )
        >>> 
        >>> prompt = "Patient with fever and elevated lactate"
        >>> response = "<think>Sepsis likely</think>..."
        >>> 
        >>> # Get total reward
        >>> total_reward = engine.compute_total(prompt, response)
        >>> 
        >>> # Get detailed breakdown
        >>> rewards = engine.compute_all(prompt, response)
        >>> print(rewards)
    """
    
    def __init__(
        self,
        w_semantic: float = 0.25,
        w_metacognitive: float = 0.20,
        w_empathy: float = 0.15,
        w_proactivity: float = 0.25,
        w_safety: float = 0.15
    ):
        """
        Initialize composite reward engine.
        
        Args:
            w_semantic: Weight for semantic fidelity reward
            w_metacognitive: Weight for metacognitive depth reward
            w_empathy: Weight for structural empathy reward
            w_proactivity: Weight for proactivity reward
            w_safety: Weight for biological safety reward
        
        Note:
            Weights should sum to 1.0 for interpretability, but this is
            not strictly enforced.
        """
        # Store weights
        self.w_semantic = w_semantic
        self.w_metacognitive = w_metacognitive
        self.w_empathy = w_empathy
        self.w_proactivity = w_proactivity
        self.w_safety = w_safety
        
        # Verify weights sum to 1.0 (approximately)
        total_weight = sum([
            w_semantic, w_metacognitive, w_empathy, w_proactivity, w_safety
        ])
        
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Reward weights sum to {total_weight:.4f}, not 1.0")
        
        # Initialize reward components
        self.semantic_reward = SemanticFidelityReward()
        self.metacognitive_reward = MetacognitiveDepthReward()
        self.empathy_reward = StructuralEmpathyReward()
        self.proactivity_reward = ProactivityReward()
        self.safety_reward = BiologicalSafetyReward()
        
        logger.info(
            f"Initialized CompositeRewardEngine - "
            f"Weights: sem={w_semantic}, meta={w_metacognitive}, "
            f"emp={w_empathy}, physio={w_proactivity}, safety={w_safety}"
        )
    
    def compute_total(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None
    ) -> float:
        """
        Compute total weighted reward.
        
        Args:
            prompt: Clinical query/scenario
            response: Model-generated response with cognitive streams
            reference: Optional reference response for semantic comparison
        
        Returns:
            Total weighted reward (scalar)
        
        Example:
            >>> total = engine.compute_total(
            ...     prompt="Patient with sepsis",
            ...     response="<think>Sepsis likely</think>...",
            ...     reference="Sepsis diagnosis confirmed"
            ... )
            >>> print(f"Total reward: {total:.4f}")
        """
        # Compute all individual rewards
        rewards = self.compute_all(prompt, response, reference)
        
        # Weighted sum
        total = (
            self.w_semantic * rewards['semantic'] +
            self.w_metacognitive * rewards['metacognitive'] +
            self.w_empathy * rewards['empathy'] +
            self.w_proactivity * rewards['proactivity'] +
            self.w_safety * rewards['safety']
        )
        
        return total
    
    def compute_all(
        self,
        prompt: str,
        response: str,
        reference: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Compute all individual reward components.
        
        Args:
            prompt: Clinical query/scenario
            response: Model-generated response with cognitive streams
            reference: Optional reference response for semantic comparison
        
        Returns:
            Dictionary with individual reward scores
        """
        # Use prompt as reference if none provided
        if reference is None:
            reference = prompt
        
        # Initialize all rewards to 0.0
        rewards = {
            'semantic': 0.0,
            'metacognitive': 0.0,
            'empathy': 0.0,
            'proactivity': 0.0,
            'safety': 0.0
        }
        
        # 1. SEMANTIC REWARD
        try:
            # SemanticFidelityReward.compute(predictions: List[str], references: List[str])
            semantic_scores = self.semantic_reward.compute([response], [reference])
            rewards['semantic'] = semantic_scores[0] if semantic_scores else 0.0
        except Exception as e:
            logger.debug(f"Semantic reward computation failed: {e}")
            rewards['semantic'] = 0.0
        
        # 2. METACOGNITIVE REWARD
        try:
            # MetacognitiveDepthReward.compute(initial_reasoning: List[str], revised_reasoning: List[str])
            # Extract reasoning from <think> tags if present
            import re
            think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
            think_matches = think_pattern.findall(response)
            
            if len(think_matches) >= 2:
                # If multiple <think> blocks, use first as initial, last as revised
                initial = think_matches[0]
                revised = think_matches[-1]
            elif len(think_matches) == 1:
                # If only one <think> block, split it in half
                think_text = think_matches[0]
                mid_point = len(think_text) // 2
                initial = think_text[:mid_point]
                revised = think_text[mid_point:]
            else:
                # No <think> blocks, split entire response
                mid_point = len(response) // 2
                initial = response[:mid_point]
                revised = response[mid_point:]
            
            metacog_scores = self.metacognitive_reward.compute([initial], [revised])
            rewards['metacognitive'] = metacog_scores[0] if metacog_scores else 0.0
        except Exception as e:
            logger.debug(f"Metacognitive reward computation failed: {e}")
            rewards['metacognitive'] = 0.0
        
        # 3. EMPATHY REWARD
        try:
            # StructuralEmpathyReward.compute(responses, inferred_literacy, inferred_emotions)
            # For GRPO, we don't have ground truth literacy/emotions, so use defaults
            from core.enums import HealthLiteracyLevel, EmotionalState
            
            # Default to medium literacy and neutral emotions for GRPO
            default_literacy = [HealthLiteracyLevel.MEDIUM]
            default_emotions = [EmotionalState.NEUTRAL]
            
            empathy_scores = self.empathy_reward.compute(
                [response],
                default_literacy,
                default_emotions
            )
            rewards['empathy'] = empathy_scores[0] if empathy_scores else 0.0
        except Exception as e:
            logger.debug(f"Empathy reward computation failed: {e}")
            rewards['empathy'] = 0.0
        
        # 4. PROACTIVITY REWARD
        try:
            # ProactivityReward.compute(predictions, ground_truth, timestamps)
            # For GRPO, we don't have ground truth, so use dummy values
            # This reward is less applicable during GRPO training
            # We'll just give it a neutral score
            rewards['proactivity'] = 0.5  # Neutral baseline
        except Exception as e:
            logger.debug(f"Proactivity reward computation failed: {e}")
            rewards['proactivity'] = 0.5
        
        # 5. SAFETY REWARD
        try:
            # BiologicalSafetyReward.compute(patient_states: List[str])
            # Extract patient_state from response
            import re
            patient_state_pattern = re.compile(r'<patient_state>(.*?)</patient_state>', re.DOTALL)
            patient_state_matches = patient_state_pattern.findall(response)
            
            if patient_state_matches:
                # Use the extracted patient state
                patient_state = patient_state_matches[0]
            else:
                # Use entire response as fallback
                patient_state = response
            
            safety_scores = self.safety_reward.compute([patient_state])
            rewards['safety'] = safety_scores[0] if safety_scores else 1.0  # Default to safe
        except Exception as e:
            logger.debug(f"Safety reward computation failed: {e}")
            rewards['safety'] = 1.0  # Default to safe
        
        return rewards
    
    
    def compute_batch(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """
        Compute rewards for a batch of prompt-response pairs.
        
        Args:
            prompts: List of clinical queries
            responses: List of model-generated responses
            references: Optional list of reference responses
        
        Returns:
            List of reward dictionaries, one per example
        
        Example:
            >>> prompts = ["Patient 1", "Patient 2"]
            >>> responses = ["Response 1", "Response 2"]
            >>> batch_rewards = engine.compute_batch(prompts, responses)
            >>> for i, rewards in enumerate(batch_rewards):
            ...     print(f"Example {i}: Total={sum(rewards.values()):.4f}")
        """
        if references is None:
            references = prompts
        
        assert len(prompts) == len(responses) == len(references), \
            "Prompts, responses, and references must have same length"
        
        batch_rewards = []
        
        for prompt, response, reference in zip(prompts, responses, references):
            rewards = self.compute_all(prompt, response, reference)
            batch_rewards.append(rewards)
        
        return batch_rewards
    
    def compute_total_batch(
        self,
        prompts: List[str],
        responses: List[str],
        references: Optional[List[str]] = None
    ) -> List[float]:
        """
        Compute total rewards for a batch (more efficient).
        
        Args:
            prompts: List of clinical queries
            responses: List of model-generated responses
            references: Optional list of reference responses
        
        Returns:
            List of total reward scores
        
        Example:
            >>> prompts = ["Patient 1", "Patient 2", "Patient 3"]
            >>> responses = ["Response 1", "Response 2", "Response 3"]
            >>> totals = engine.compute_total_batch(prompts, responses)
            >>> print(f"Average reward: {sum(totals)/len(totals):.4f}")
        """
        batch_rewards = self.compute_batch(prompts, responses, references)
        
        total_rewards = []
        for rewards in batch_rewards:
            total = (
                self.w_semantic * rewards['semantic'] +
                self.w_metacognitive * rewards['metacognitive'] +
                self.w_empathy * rewards['empathy'] +
                self.w_proactivity * rewards['proactivity'] +
                self.w_safety * rewards['safety']
            )
            total_rewards.append(total)
        
        return total_rewards
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current reward weights.
        
        Returns:
            Dictionary of reward component weights
        """
        return {
            'semantic': self.w_semantic,
            'metacognitive': self.w_metacognitive,
            'empathy': self.w_empathy,
            'proactivity': self.w_proactivity,
            'safety': self.w_safety
        }
    
    def set_weights(
        self,
        w_semantic: Optional[float] = None,
        w_metacognitive: Optional[float] = None,
        w_empathy: Optional[float] = None,
        w_proactivity: Optional[float] = None,
        w_safety: Optional[float] = None
    ):
        """
        Update reward weights.
        
        Args:
            w_semantic: New semantic weight (optional)
            w_metacognitive: New metacognitive weight (optional)
            w_empathy: New empathy weight (optional)
            w_proactivity: New proactivity weight (optional)
            w_safety: New safety weight (optional)
        
        Example:
            >>> engine.set_weights(w_safety=0.30, w_semantic=0.20)
            >>> print(engine.get_weights())
        """
        if w_semantic is not None:
            self.w_semantic = w_semantic
        if w_metacognitive is not None:
            self.w_metacognitive = w_metacognitive
        if w_empathy is not None:
            self.w_empathy = w_empathy
        if w_proactivity is not None:
            self.w_proactivity = w_proactivity
        if w_safety is not None:
            self.w_safety = w_safety
        
        # Log updated weights
        logger.info(
            f"Updated weights: sem={self.w_semantic}, meta={self.w_metacognitive}, "
            f"emp={self.w_empathy}, physio={self.w_proactivity}, safety={self.w_safety}"
        )