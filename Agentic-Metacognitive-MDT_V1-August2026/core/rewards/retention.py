"""
core/rewards/retention.py -- R_retention: longitudinal context retention.

Ported verbatim from the source repo's training/rewards.py.
"""

from __future__ import annotations

import numpy as np

from core.parsing import ParsedStreams
from core.rewards._encoder import get_sentence_encoder


# ---------------------------------------------------------------------------
# R_retention: longitudinal context retention
# ---------------------------------------------------------------------------

def reward_context_retention(parsed: ParsedStreams, must_mention_facts: list[str]) -> float:
    """Checks whether key facts established earlier in a multi-turn / long
    admission context (e.g. "patient has a documented penicillin allergy")
    are correctly retained and not contradicted in <patient_state>.
    `must_mention_facts` is a list of short strings the caller expects to
    see reflected (not necessarily verbatim -- checked via substring OR
    embedding similarity fallback).
    """
    if parsed.patient_state is None or not must_mention_facts:
        return 0.0 if must_mention_facts else 1.0
    text = parsed.patient_state.lower()
    hits = 0
    encoder = get_sentence_encoder()
    for fact in must_mention_facts:
        if fact.lower() in text:
            hits += 1
            continue
        emb = encoder.encode([fact, parsed.patient_state], convert_to_numpy=True)
        cos_sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8))
        if cos_sim > 0.6:
            hits += 1
    return hits / len(must_mention_facts)
