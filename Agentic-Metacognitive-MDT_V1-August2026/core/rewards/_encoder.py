"""
core/rewards/_encoder.py

Shared sentence-encoder resource used by the embedding-based reward
components (semantic fidelity, metacognitive self-correction, context
retention). Split out of the source repo's single training/rewards.py so
each reward component can live in its own module without three separate
copies of the lazy-loading singleton, and so importing e.g.
core.rewards.format never touches this file at all.

The lazy import inside get_sentence_encoder is load-bearing and preserved
verbatim from the source -- see its comment.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared resources (loaded once, reused across reward calls)
# ---------------------------------------------------------------------------

_SENTENCE_ENCODER: Optional["SentenceTransformer"] = None


def get_sentence_encoder(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    # Lazy import: SentenceTransformer (and the torch it pulls in) is only
    # actually needed by the reward functions that compute embedding
    # similarity (semantic fidelity, context retention, metacognitive self-
    # correction) -- importing it at module level would force every OTHER
    # reward function in this file (reward_format, reward_physio_grounding,
    # reward_empathy, reward_tool_call, reward_forecast_accuracy -- all pure
    # regex/string/numpy logic) to require a full torch install just to be
    # imported at all, which is exactly the testability problem caught and
    # fixed in models/stream_parsing.py; applying the same fix here.
    from sentence_transformers import SentenceTransformer
    global _SENTENCE_ENCODER
    if _SENTENCE_ENCODER is None:
        _SENTENCE_ENCODER = SentenceTransformer(model_name)
        # Held fixed, never fine-tuned jointly with the policy -- see proposal's
        # reward-hacking mitigation discussion (R_meta section).
        for p in _SENTENCE_ENCODER.parameters():
            p.requires_grad = False
    return _SENTENCE_ENCODER
