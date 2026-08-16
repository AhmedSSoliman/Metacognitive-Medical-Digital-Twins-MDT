"""
core/rewards/boundary.py -- R_bound: topological hypergraph bounds.

Ported verbatim from the source repo's training/rewards.py. The checker is
still passed IN rather than imported (see the docstring) -- the concrete
implementations live in core/hypergraph/verification.py.
"""

from __future__ import annotations

from core.parsing import ParsedStreams


# ---------------------------------------------------------------------------
# R_bound: topological hypergraph bounds (interim rule-based OR learned hypergraph)
# ---------------------------------------------------------------------------

def reward_hypergraph_bound(parsed: ParsedStreams, hypergraph_checker) -> float:
    """`hypergraph_checker` is an object exposing `.check(patient_state_text) -> float in [0,1]`,
    implemented in hypergraph/verification.py. Passed in rather than imported directly so
    Phase 2 can run against either the interim rule-based constraint or the Phase 3
    data-derived hypergraph without changing this file.
    """
    if parsed.patient_state is None:
        return 0.0
    return float(hypergraph_checker.check(parsed.patient_state))
