"""
core/rewards/diagnostic.py -- R_diagnostic: asymmetric clinical diagnostic
accuracy.

Ported verbatim from the source repo's training/rewards.py.

DEVIATION FROM THE TARGET TREE (documented deliberately): the target
structure listed nine reward modules (format, semantic, metacognitive,
empathy, physiological, boundary, tool_use, retention, forecast). The source
repo's training/rewards.py actually defines TEN reward components -- this
one, R_diagnostic, was added after the docstring's "eight reward components"
header was written and is wired into compute_total_reward with the highest
non-safety weight (w_diagnostic=1.5). Dropping it to match the target tree
would have silently changed the total-reward computation, so it gets its own
module here following the same one-component-per-file pattern.
"""

from __future__ import annotations

from core.parsing import ParsedStreams


# ---------------------------------------------------------------------------
# R_diagnostic: asymmetric clinical diagnostic accuracy
# ---------------------------------------------------------------------------

def reward_clinical_diagnostic_accuracy(parsed: ParsedStreams, reference_patient_state: str) -> float:
    """Asymmetric diagnostic alarm alignment. Penalizes False Negatives (missing
    deterioration indicators in patient state description) severely (-2.0) and
    rewards True Positives (correctly matching critical state indicators).
    """
    if parsed.patient_state is None or not reference_patient_state:
        return 0.0
    pred_clean = parsed.patient_state.lower()
    ref_clean = reference_patient_state.lower()
    
    critical_keywords = [
        "surge", "critical", "shock", "deterioration", "severe", "acute", 
        "worsening", "decline", "hypoxia", "sepsis", "failure", "unstable", 
        "urgent", "emergency", "arrest", "infarction"
    ]
    
    gt_critical = any(w in ref_clean for w in critical_keywords)
    pred_critical = any(w in pred_clean for w in critical_keywords)
    
    if gt_critical and pred_critical:
        # True Positive
        return 1.0
    elif not gt_critical and not pred_critical:
        # True Negative
        return 0.2
    elif not gt_critical and pred_critical:
        # False Positive (neutral safe exploration)
        return 0.5
    else:
        # False Negative (severe penalty)
        return -2.0
