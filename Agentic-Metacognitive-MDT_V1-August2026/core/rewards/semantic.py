"""
core/rewards/semantic.py -- R_sem: semantic content fidelity.

Ported verbatim from the source repo's training/rewards.py. Imports rewritten
to core.parsing / core.rewards._encoder. Needs sentence-transformers (and
therefore torch) only at CALL time, never at import time -- see
core/rewards/_encoder.py.
"""

from __future__ import annotations

import numpy as np

from core.parsing import ParsedStreams
from core.rewards._encoder import get_sentence_encoder


# ---------------------------------------------------------------------------
# R_sem: semantic content fidelity (reference-based, e.g. vs. clinician-written target)
# ---------------------------------------------------------------------------

def reward_semantic_fidelity(parsed: ParsedStreams, reference_patient_state: str) -> float:
    if parsed.patient_state is None or not reference_patient_state:
        return 0.0
    encoder = get_sentence_encoder()
    emb = encoder.encode([parsed.patient_state, reference_patient_state], convert_to_numpy=True)
    cos_sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8))
    return max(0.0, cos_sim)
