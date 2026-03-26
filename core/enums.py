"""Enumerations for Medical Digital Twin system."""

from enum import Enum


class HealthLiteracyLevel(Enum):
    """Health literacy classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EmotionalState(Enum):
    """Emotional state classification."""
    CALM = "calm"
    ANXIOUS = "anxious"
    DISTRESSED = "distressed"