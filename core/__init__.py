"""Core modules for Medical Digital Twin."""

from .enums import HealthLiteracyLevel, EmotionalState
from .cognitive_streams import CognitiveStreams, CognitiveStreamParser

# Keep heavy optional modules lazy to avoid importing torch/cuda at package import time.
try:
    from .theory_of_mind import TheoryOfMindModule
except Exception:  # pragma: no cover - optional runtime dependency
    TheoryOfMindModule = None

__all__ = [
    'HealthLiteracyLevel',
    'EmotionalState',
    'CognitiveStreams',
    'CognitiveStreamParser',
    'TheoryOfMindModule'
]