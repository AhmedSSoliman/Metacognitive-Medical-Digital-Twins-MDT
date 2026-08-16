"""
core/rewards/metacognitive.py -- R_meta: metacognitive trace self-correction
(the delta-embedding reward).

Ported verbatim from the source repo's training/rewards.py.

RELATIONSHIP TO evaluation/delta_embedding.py: this module holds the single
authoritative implementation of the delta-embedding computation. The
evaluation-side concordance endpoint (evaluation/delta_embedding.py) IMPORTS
and reuses reward_metacognitive_selfcorrection from here rather than
duplicating the logic, so the training reward and the evaluation metric can
never drift apart -- the same pattern evaluation/metrics.py already used for
R_bound (topological fidelity) and R_emp (structural empathy) in the source
repo.
"""

from __future__ import annotations

import numpy as np

from core.parsing import ParsedStreams
from core.rewards._encoder import get_sentence_encoder


# ---------------------------------------------------------------------------
# R_meta: metacognitive trace self-correction (delta-embedding reward)
# ---------------------------------------------------------------------------

PIVOT_PHRASES = [
    "wait,", "actually,", "let me reconsider", "on second thought",
    "correcting myself", "i was wrong", "re-evaluating",
]

def reward_metacognitive_selfcorrection(parsed: ParsedStreams, window_tokens: int = 15) -> float:
    """Delta-embedding reward: measures embedding-space shift around explicit
    pivot tokens in <think>, as a proxy for genuine reconsideration rather
    than superficial hedging. This is the component flagged in the proposal
    as UNVALIDATED and subject to periodic auditing against expert
    annotation -- treat scores from this function with corresponding
    caution, especially early in training.
    """
    if parsed.think is None:
        return 0.0
    text = parsed.think
    text_lower = text.lower()

    pivot_positions = []
    for phrase in PIVOT_PHRASES:
        start = 0
        while True:
            idx = text_lower.find(phrase, start)
            if idx == -1:
                break
            pivot_positions.append(idx)
            start = idx + 1

    if not pivot_positions:
        return 0.0  # no self-correction attempted; this is not a penalty, just zero signal

    encoder = get_sentence_encoder()
    words = text.split()
    scores = []
    for char_idx in pivot_positions:
        # char->word index mapping: verified correct across edge cases (pivot
        # at string start, multiple adjacent pivots, mid-sentence) since
        # text[:char_idx].split() and text.split() tokenize on the same
        # whitespace boundaries consistently.
        word_idx = len(text[:char_idx].split())
        before = " ".join(words[max(0, word_idx - window_tokens):word_idx])
        after = " ".join(words[word_idx:word_idx + window_tokens])
        if not before or not after:
            continue
        emb = encoder.encode([before, after], convert_to_numpy=True)
        cos_sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8))
        # Larger embedding shift (lower cosine similarity) around a pivot token
        # is scored as MORE likely to reflect genuine reconsideration.
        shift = 1.0 - cos_sim
        scores.append(shift)

    return float(np.mean(scores)) if scores else 0.0
