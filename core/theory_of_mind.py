"""
Theory of Mind module for inferring user belief states.

Implements hierarchical neural network for literacy and emotion inference.
"""

import torch
import torch.nn as nn
from typing import Tuple

from core.enums import HealthLiteracyLevel, EmotionalState


class TheoryOfMindModule(nn.Module):
    """Neural Theory of Mind module for inferring user belief states."""
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_literacy_levels: int = 3,
        num_emotional_states: int = 3
    ):
        """Initialize Theory of Mind module."""
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Literacy level classifier
        self.literacy_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_literacy_levels)
        )
        
        # Emotional state classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotional_states)
        )
        
        # Belief state encoder
        self.belief_encoder = nn.Sequential(
            nn.Linear(hidden_dim + num_literacy_levels + num_emotional_states, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, user_embedding: torch.Tensor) -> dict:
        """Forward pass to infer user belief state."""
        # Classify health literacy
        literacy_logits = self.literacy_classifier(user_embedding)
        literacy_probs = torch.softmax(literacy_logits, dim=-1)
        
        # Classify emotional state
        emotion_logits = self.emotion_classifier(user_embedding)
        emotion_probs = torch.softmax(emotion_logits, dim=-1)
        
        # Encode combined belief state
        combined = torch.cat([user_embedding, literacy_probs, emotion_probs], dim=-1)
        belief_state = self.belief_encoder(combined)
        
        return {
            'literacy_logits': literacy_logits,
            'literacy_probs': literacy_probs,
            'emotion_logits': emotion_logits,
            'emotion_probs': emotion_probs,
            'belief_state': belief_state
        }
    
    def infer_literacy_level(
        self,
        user_embedding: torch.Tensor
    ) -> Tuple[HealthLiteracyLevel, float]:
        """Infer health literacy level with confidence."""
        with torch.no_grad():
            logits = self.literacy_classifier(user_embedding)
            probs = torch.softmax(logits, dim=-1)
            
            predicted_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_idx].item()
            
            levels = [HealthLiteracyLevel.LOW, HealthLiteracyLevel.MEDIUM, HealthLiteracyLevel.HIGH]
            
            return levels[predicted_idx], confidence
    
    def infer_emotional_state(
        self,
        user_embedding: torch.Tensor
    ) -> Tuple[EmotionalState, float]:
        """Infer emotional state with confidence."""
        with torch.no_grad():
            logits = self.emotion_classifier(user_embedding)
            probs = torch.softmax(logits, dim=-1)
            
            predicted_idx = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, predicted_idx].item()
            
            states = [EmotionalState.CALM, EmotionalState.ANXIOUS, EmotionalState.DISTRESSED]
            
            return states[predicted_idx], confidence