"""
core/rewards/physiological.py -- R_physio: physiological vocabulary grounding.

Ported verbatim from the source repo's training/rewards.py. Pure string logic.
"""

from __future__ import annotations

from core.parsing import ParsedStreams


# ---------------------------------------------------------------------------
# R_physio: physiological vocabulary grounding
# ---------------------------------------------------------------------------

_PHYSIO_TERMS = {
    "tachycardia", "bradycardia", "hypotension", "hypertension", "hypoxia",
    "hyperlactatemia", "lactate", "map", "mean arterial pressure", "spo2",
    "respiratory rate", "tachypnea", "oliguria", "anuria", "afib", "sepsis",
    "shock", "vasopressor", "creatinine", "bilirubin", "leukocytosis",
    "thrombocytopenia", "acidosis", "alkalosis", "hyperkalemia", "hypokalemia",
}

def reward_physio_grounding(parsed: ParsedStreams) -> float:
    if parsed.patient_state is None:
        return 0.0
    text = parsed.patient_state.lower()
    hits = sum(1 for term in _PHYSIO_TERMS if term in text)
    # Normalize against a soft target of ~3 grounded terms per patient_state block
    return min(1.0, hits / 3.0)
