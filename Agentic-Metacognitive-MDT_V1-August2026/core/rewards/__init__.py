"""
core/rewards

The reward components for Phase 2 multi-objective GRPO alignment, one module
per component (split out of the source repo's single training/rewards.py).

Everything the old `from training.rewards import ...` call sites needed is
re-exported here, so `from core.rewards import reward_format, ...` works as a
drop-in replacement. Note this convenience import pulls in EVERY component
module, including the three that call sentence-transformers at runtime --
importing the sub-module directly (e.g. `from core.rewards.format import
reward_format`) stays the lightest path, though no module in this package
imports torch at IMPORT time (see core/rewards/_encoder.py).
"""

from core.rewards._encoder import get_sentence_encoder
from core.rewards.boundary import reward_hypergraph_bound
from core.rewards.composite import RewardWeights, compute_total_reward
from core.rewards.diagnostic import reward_clinical_diagnostic_accuracy
from core.rewards.empathy import (
    compute_flesch_kincaid_grade,
    count_syllables,
    reward_empathy,
)
from core.rewards.forecast import reward_forecast_accuracy
from core.rewards.format import reward_format
from core.rewards.metacognitive import (
    PIVOT_PHRASES,
    reward_metacognitive_selfcorrection,
)
from core.rewards.physiological import reward_physio_grounding
from core.rewards.retention import reward_context_retention
from core.rewards.semantic import reward_semantic_fidelity
from core.rewards.theory_of_mind import reward_theory_of_mind
from core.rewards.tool_use import reward_tool_call

__all__ = [
    "reward_theory_of_mind",
    "get_sentence_encoder",
    "reward_format",
    "reward_semantic_fidelity",
    "reward_physio_grounding",
    "reward_hypergraph_bound",
    "reward_tool_call",
    "reward_empathy",
    "reward_metacognitive_selfcorrection",
    "reward_context_retention",
    "reward_forecast_accuracy",
    "reward_clinical_diagnostic_accuracy",
    "RewardWeights",
    "compute_total_reward",
    "count_syllables",
    "compute_flesch_kincaid_grade",
    "PIVOT_PHRASES",
]
