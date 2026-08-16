"""
evaluation/topological.py -- Endpoint 1: Topological Fidelity.

Ported verbatim from the source repo's evaluation/metrics.py
(../Agentic-DT_V1-July/evaluation/metrics.py). Imports rewritten:
parse_streams from core.parsing (not the torch-importing model wrapper), and
reward_hypergraph_bound from core.rewards.boundary.
"""

from __future__ import annotations

import numpy as np

from core.parsing import parse_streams
from core.rewards.boundary import reward_hypergraph_bound


def evaluate_topological_fidelity(generations: list[str], hypergraph_checker) -> float:
    """Endpoint 1: how often does the model's <patient_state> claim a
    combination of abnormalities that IS supported by the (clinically
    reviewed) hypergraph, versus claiming something physiologically
    ungrounded? Reuses reward_hypergraph_bound directly rather than
    reimplementing the check, so this metric and the R_bound training
    reward are guaranteed to agree on what "grounded" means.
    """
    scores = []
    for gen in generations:
        parsed = parse_streams(gen)
        scores.append(reward_hypergraph_bound(parsed, hypergraph_checker))
    return float(np.mean(scores)) if scores else 0.0
