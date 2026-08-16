"""
evaluation/communication.py -- Endpoint 5: Structural Empathy /
communication adaptation scoring.

Ported verbatim from the source repo's evaluation/metrics.py
(evaluate_structural_empathy). The target tree names this file
"communication.py" and describes it as "Communication adaptation scoring";
the source function is named evaluate_structural_empathy after the
proposal's Endpoint 5 label. The original name is kept (renaming it would
break the call in evaluation/report.py's run_full_evaluation and diverge from
the proposal's own terminology) and `evaluate_communication_adaptation` is
provided as an alias matching the target-tree naming.
"""

from __future__ import annotations

import numpy as np

from core.parsing import parse_streams
from core.rewards.empathy import reward_empathy


def evaluate_structural_empathy(generations: list[str], recipient_types: list[str]) -> float:
    """Endpoint 5: how well <user_belief>-driven framing adapts to the
    intended recipient (clinician vs. patient vs. family), reusing
    reward_empathy directly so this metric and the R_emp training reward
    stay consistent with each other, same rationale as Endpoint 1 above.
    """
    scores = []
    for gen, recipient in zip(generations, recipient_types):
        parsed = parse_streams(gen)
        scores.append(reward_empathy(parsed, recipient))
    return float(np.mean(scores)) if scores else 0.0


# Alias matching the target tree's "communication adaptation scoring" naming.
# Same function, no wrapper behavior -- see this module's docstring.
evaluate_communication_adaptation = evaluate_structural_empathy
