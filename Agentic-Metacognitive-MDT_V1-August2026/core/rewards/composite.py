"""
core/rewards/composite.py

Weighted combination of every reward component into a single total.

Ported verbatim from the "Aggregate reward" section of the source repo's
training/rewards.py (RewardWeights + compute_total_reward). Only the imports
changed: each component now comes from its own module under core.rewards
instead of being defined above in the same file.

NOTE ON "curriculum sampling" (target-tree spec): the target structure
described this file as "Weighted combination, curriculum sampling". Only the
weighted combination exists in the source repo. There is no curriculum
sampling over reward components anywhere (grep for 'curriculum' across the
source finds only training/sft_trainer.py's TWO-STAGE SFT CURRICULUM -- the
--soft_alignment vignette-fraction ramp -- which is a DATA-MIX curriculum for
Phase 1 SFT, not a reward curriculum, and is preserved in training/sft.py).
If a reward-component curriculum is added later, this is where it belongs.

The original module docstring of training/rewards.py is preserved here since
this file is the aggregate entry point:

    The eight reward components for Phase 2 multi-objective GRPO alignment.
    Each function takes a parsed generation (see models.multi_stream.ParsedStreams)
    plus whatever reference data it needs, and returns a scalar in roughly [0, 1]
    (a few are unbounded log-likelihood-style scores, noted below).

    These are reference implementations meant to be correct and runnable, not
    final tuned versions -- exact thresholds and embedding models should be
    validated against a held-out sample with clinician review before trusting
    scores at scale (this mirrors the proposal's own emphasis on periodic
    auditing of R_meta specifically).

(The header says "eight"; the file grew to ten components over time -- see
core/rewards/diagnostic.py's note. An eleventh, R_tom, was added 2026-08-13
-- see core/rewards/theory_of_mind.py's module docstring for why R_emp's
style-only check of <user_belief> wasn't sufficient on its own.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.parsing import parse_streams
from core.rewards.boundary import reward_hypergraph_bound
from core.rewards.diagnostic import reward_clinical_diagnostic_accuracy
from core.rewards.empathy import reward_empathy
from core.rewards.forecast import reward_forecast_accuracy
from core.rewards.format import reward_format
from core.rewards.metacognitive import reward_metacognitive_selfcorrection
from core.rewards.physiological import reward_physio_grounding
from core.rewards.retention import reward_context_retention
from core.rewards.semantic import reward_semantic_fidelity
from core.rewards.theory_of_mind import reward_theory_of_mind
from core.rewards.tool_use import reward_tool_call

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregate reward
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    w_format: float = 1.0
    w_sem: float = 1.0
    w_physio: float = 0.5
    w_bound: float = 2.0     # weighted higher: this is the safety-critical term
    w_tool: float = 0.5
    w_emp: float = 1.0
    w_meta: float = 0.75     # weighted lower initially given it is unvalidated
    w_retention: float = 1.0
    w_forecast: float = 1.0
    w_diagnostic: float = 1.5 # high safety weight for early detection alignment
    w_tom: float = 0.5        # weighted lower initially, same rationale as w_meta: the
                               # synthetic knowledge-state ground truth it's checked against
                               # is unvalidated against real clinician/patient judgment


def compute_total_reward(
    generated_text: str,
    reference_patient_state: str,
    hypergraph_checker,
    recipient_type: str,
    must_mention_facts: list[str],
    true_future_values: Optional[dict[str, float]] = None,
    recipient_knows: Optional[list[str]] = None,
    recipient_does_not_know: Optional[list[str]] = None,
    weights: RewardWeights = RewardWeights(),
) -> dict:
    parsed = parse_streams(generated_text)
    true_future_values = true_future_values or {}

    components = {
        "R_format": reward_format(generated_text),
        "R_sem": reward_semantic_fidelity(parsed, reference_patient_state),
        "R_physio": reward_physio_grounding(parsed),
        "R_bound": reward_hypergraph_bound(parsed, hypergraph_checker),
        "R_tool": reward_tool_call(generated_text),
        "R_emp": reward_empathy(parsed, recipient_type),
        "R_meta": reward_metacognitive_selfcorrection(parsed),
        "R_retention": reward_context_retention(parsed, must_mention_facts),
        "R_forecast": reward_forecast_accuracy(parsed, true_future_values),
        "R_diagnostic": reward_clinical_diagnostic_accuracy(parsed, reference_patient_state),
        "R_tom": reward_theory_of_mind(parsed, recipient_knows, recipient_does_not_know),
    }

    total = (
        weights.w_format * components["R_format"]
        + weights.w_sem * components["R_sem"]
        + weights.w_physio * components["R_physio"]
        + weights.w_bound * components["R_bound"]
        + weights.w_tool * components["R_tool"]
        + weights.w_emp * components["R_emp"]
        + weights.w_meta * components["R_meta"]
        + weights.w_retention * components["R_retention"]
        + weights.w_forecast * components["R_forecast"]
        + weights.w_diagnostic * components["R_diagnostic"]
        + weights.w_tom * components["R_tom"]
    )
    components["total"] = total
    return components


if __name__ == "__main__":
    # Smoke test with a dummy hypergraph checker
    class _DummyChecker:
        def check(self, text):
            return 0.8

    example = (
        "<think>HR is climbing, wait, actually the MAP trend is more concerning here.</think>"
        "<patient_state>Tachycardic at 118, MAP 58, lactate rising to 3.1 -- concerning for early "
        "septic shock.</patient_state>"
        "<forecast>MAP_6h: 55 [48-62]\nlactate_6h: 3.8 [3.0-4.6]</forecast>"
        "<user_belief>This will be read by the bedside nurse; keep it concrete and action-oriented.</user_belief>"
    )
    result = compute_total_reward(
        generated_text=example,
        reference_patient_state="Patient is tachycardic and hypotensive with rising lactate, "
                                 "consistent with early septic shock.",
        hypergraph_checker=_DummyChecker(),
        recipient_type="clinician",
        must_mention_facts=["septic shock"],
        true_future_values={"MAP_6h": 53, "lactate_6h": 4.1},  # both fall inside the predicted intervals
    )
    for k, v in result.items():
        print(f"{k}: {v:.3f}")
    assert result["R_forecast"] == 1.0, "Both true values fall inside the predicted intervals -- expected perfect score"
    print("Forecast reward smoke test passed.")
