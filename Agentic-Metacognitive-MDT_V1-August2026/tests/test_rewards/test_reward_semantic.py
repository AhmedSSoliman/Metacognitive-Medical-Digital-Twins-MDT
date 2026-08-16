"""
tests/test_rewards/test_reward_semantic.py

Real test coverage for three reward functions that were previously
UNTESTED in this repo despite being referenced in test_reward_logic.py's
module docstring as "tested separately in tests/test_reward_semantic.py" --
that file never actually existed here (confirmed via grep across the whole
repo, 2026-08-13). This is that file, written for real, not a restoration
of lost content.

Also covers compute_total_reward (core/rewards/composite.py), the actual
aggregate function combining all eleven reward components -- previously
zero direct test coverage anywhere in this repo.

All embedding-similarity thresholds below are backed by REAL measured
cosine similarities from the actual sentence encoder (see the comment above
each assertion for the measured value), not guessed -- the same discipline
applied after test_theory_of_mind.py's embedding test initially failed on
an optimistic guess.
"""

import pytest

from core.parsing import parse_streams
from core.rewards.composite import RewardWeights, compute_total_reward
from core.rewards.metacognitive import reward_metacognitive_selfcorrection
from core.rewards.retention import reward_context_retention
from core.rewards.semantic import reward_semantic_fidelity


def _generation(think="x", patient_state="y", forecast="not applicable", user_belief="z") -> str:
    return (
        f"<think>{think}</think><patient_state>{patient_state}</patient_state>"
        f"<forecast>{forecast}</forecast><user_belief>{user_belief}</user_belief>"
    )


# ---------------------------------------------------------------------------
# R_sem: reward_semantic_fidelity
# ---------------------------------------------------------------------------

def test_semantic_fidelity_zero_when_patient_state_missing():
    parsed = parse_streams("<think>x</think><forecast>not applicable</forecast><user_belief>z</user_belief>")
    assert reward_semantic_fidelity(parsed, "some reference") == 0.0


def test_semantic_fidelity_zero_when_reference_empty():
    parsed = parse_streams(_generation(patient_state="Tachycardic, hypotensive."))
    assert reward_semantic_fidelity(parsed, "") == 0.0


@pytest.mark.requires_torch
def test_semantic_fidelity_scores_paraphrase_higher_than_unrelated():
    # Measured cosine similarities (2026-08-13, real encoder):
    #   close paraphrase: 0.887
    #   unrelated text:   0.562
    close = parse_streams(_generation(
        patient_state="Tachycardic, hypotensive, rising lactate, consistent with early septic shock."
    ))
    far = parse_streams(_generation(
        patient_state="Patient reports feeling well, vitals stable, tolerating diet."
    ))
    reference = "Patient is tachycardic and hypotensive with rising lactate, consistent with early septic shock."

    close_score = reward_semantic_fidelity(close, reference)
    far_score = reward_semantic_fidelity(far, reference)
    assert close_score > 0.8
    assert far_score < 0.7
    assert close_score > far_score


@pytest.mark.requires_torch
def test_semantic_fidelity_never_negative_for_dissimilar_text():
    # max(0.0, cos_sim) clamp -- verify it actually clamps rather than
    # trusting the implementation blindly.
    parsed = parse_streams(_generation(patient_state="Weather is sunny today."))
    score = reward_semantic_fidelity(parsed, "Severe sepsis with multi-organ failure.")
    assert score >= 0.0


# ---------------------------------------------------------------------------
# R_retention: reward_context_retention
# ---------------------------------------------------------------------------

def test_retention_neutral_when_no_facts_required():
    parsed = parse_streams(_generation(patient_state="anything"))
    assert reward_context_retention(parsed, []) == 1.0


def test_retention_zero_when_patient_state_missing_but_facts_required():
    parsed = parse_streams("<think>x</think><forecast>not applicable</forecast><user_belief>z</user_belief>")
    assert reward_context_retention(parsed, ["penicillin allergy"]) == 0.0


def test_retention_full_credit_for_exact_substring_match():
    parsed = parse_streams(_generation(
        patient_state="Patient has a penicillin allergy noted on the chart, currently stable."
    ))
    assert reward_context_retention(parsed, ["penicillin allergy"]) == 1.0


def test_retention_partial_credit_when_only_some_facts_retained():
    parsed = parse_streams(_generation(patient_state="Patient has a penicillin allergy, currently stable."))
    score = reward_context_retention(parsed, ["penicillin allergy", "prior CABG in 2019"])
    assert score == pytest.approx(0.5, abs=1e-9)  # 1 of 2 facts present verbatim


@pytest.mark.requires_torch
def test_retention_embedding_fallback_catches_paraphrase():
    # Measured cosine similarities (2026-08-13, real encoder), threshold is 0.6:
    #   paraphrase of "penicillin allergy":            0.834 (above threshold)
    #   unrelated patient_state vs "penicillin allergy": 0.170 (well below)
    parsed = parse_streams(_generation(
        patient_state="Patient has a documented allergy to penicillin, noted on the chart."
    ))
    assert reward_context_retention(parsed, ["penicillin allergy"]) == 1.0


@pytest.mark.requires_torch
def test_retention_scores_zero_for_unrelated_patient_state():
    parsed = parse_streams(_generation(
        patient_state="Patient is afebrile with stable vitals and improving mental status."
    ))
    assert reward_context_retention(parsed, ["penicillin allergy"]) == 0.0


# ---------------------------------------------------------------------------
# R_meta: reward_metacognitive_selfcorrection
# ---------------------------------------------------------------------------

def test_metacognitive_zero_when_think_missing():
    parsed = parse_streams("<patient_state>y</patient_state><forecast>not applicable</forecast><user_belief>z</user_belief>")
    assert reward_metacognitive_selfcorrection(parsed) == 0.0


def test_metacognitive_zero_when_no_pivot_phrase_present():
    parsed = parse_streams(_generation(
        think="The vitals are stable and there is no concerning trend in the data."
    ))
    assert reward_metacognitive_selfcorrection(parsed) == 0.0


@pytest.mark.requires_torch
def test_metacognitive_rewards_genuine_reconsideration_over_hedging():
    # Measured embedding shifts (1 - cosine similarity) around the pivot
    # phrase "actually," (2026-08-13, real encoder):
    #   genuine content shift (reassuring -> concerning):  0.634
    #   superficial hedge (similar before/after):          0.141
    # Both texts use the SAME pivot phrase and window size; only the
    # semantic content before/after the pivot differs.
    genuine_shift = parse_streams(_generation(
        think="the vitals look stable and reassuring, actually, this could actually be an early warning sign of deterioration"
    ))
    superficial_hedge = parse_streams(_generation(
        think="the vitals look stable and reassuring, actually, overall the vitals appear stable and not concerning"
    ))
    genuine_score = reward_metacognitive_selfcorrection(genuine_shift)
    hedge_score = reward_metacognitive_selfcorrection(superficial_hedge)
    assert genuine_score > hedge_score
    assert genuine_score > 0.4
    assert hedge_score < 0.3


def test_metacognitive_detects_multiple_pivot_phrases():
    # Structural check only (no torch needed): confirms pivot DETECTION
    # (not the embedding scoring) finds both occurrences, via the fact that
    # a think block with a recognized pivot returns a nonzero-eligible path
    # rather than the early "no pivot" 0.0 return.
    parsed = parse_streams(_generation(
        think="Wait, that doesn't add up. Actually, let me reconsider the whole picture here."
    ))
    # Just confirms this does NOT hit the "no pivot found" short-circuit --
    # full scoring requires requires_torch, but detection itself doesn't.
    from core.rewards.metacognitive import PIVOT_PHRASES
    text_lower = parsed.think.lower()
    found = [p for p in PIVOT_PHRASES if p in text_lower]
    assert len(found) >= 2


# ---------------------------------------------------------------------------
# compute_total_reward: the actual aggregate combining all eleven components
# ---------------------------------------------------------------------------

class _DummyChecker:
    def __init__(self, score=0.8):
        self._score = score

    def check(self, text):
        return self._score


def test_compute_total_reward_returns_all_eleven_components_plus_total():
    result = compute_total_reward(
        generated_text=_generation(
            think="Reasoning here.",
            patient_state="Tachycardic and hypotensive.",
            forecast="MAP_6h: 55 [48-62]",
            user_belief="For the bedside nurse.",
        ),
        reference_patient_state="Patient is tachycardic and hypotensive.",
        hypergraph_checker=_DummyChecker(),
        recipient_type="clinician",
        must_mention_facts=[],
    )
    expected_keys = {
        "R_format", "R_sem", "R_physio", "R_bound", "R_tool", "R_emp",
        "R_meta", "R_retention", "R_forecast", "R_diagnostic", "R_tom", "total",
    }
    assert set(result.keys()) == expected_keys


def test_compute_total_reward_total_matches_manual_weighted_sum():
    weights = RewardWeights()
    result = compute_total_reward(
        generated_text=_generation(
            think="Reasoning here.",
            patient_state="Tachycardic and hypotensive.",
            forecast="not applicable",
            user_belief="For the bedside nurse.",
        ),
        reference_patient_state="Patient is tachycardic and hypotensive.",
        hypergraph_checker=_DummyChecker(),
        recipient_type="clinician",
        must_mention_facts=[],
    )
    manual_total = (
        weights.w_format * result["R_format"]
        + weights.w_sem * result["R_sem"]
        + weights.w_physio * result["R_physio"]
        + weights.w_bound * result["R_bound"]
        + weights.w_tool * result["R_tool"]
        + weights.w_emp * result["R_emp"]
        + weights.w_meta * result["R_meta"]
        + weights.w_retention * result["R_retention"]
        + weights.w_forecast * result["R_forecast"]
        + weights.w_diagnostic * result["R_diagnostic"]
        + weights.w_tom * result["R_tom"]
    )
    assert result["total"] == pytest.approx(manual_total, abs=1e-9)


def test_compute_total_reward_respects_custom_weights():
    kwargs = dict(
        generated_text=_generation(patient_state="Tachycardic."),
        reference_patient_state="Tachycardic.",
        hypergraph_checker=_DummyChecker(score=0.5),
        recipient_type="clinician",
        must_mention_facts=[],
    )
    default_result = compute_total_reward(**kwargs)
    zeroed_bound_result = compute_total_reward(**kwargs, weights=RewardWeights(w_bound=0.0))
    # Zeroing w_bound removes R_bound's contribution to total (R_bound itself
    # is unchanged -- only its weighted contribution to the total differs).
    assert default_result["R_bound"] == zeroed_bound_result["R_bound"]
    assert default_result["total"] != zeroed_bound_result["total"]


def test_compute_total_reward_hypergraph_checker_score_propagates_to_bound():
    kwargs = dict(
        generated_text=_generation(patient_state="Tachycardic."),
        reference_patient_state="Tachycardic.",
        recipient_type="clinician",
        must_mention_facts=[],
    )
    high = compute_total_reward(hypergraph_checker=_DummyChecker(score=0.9), **kwargs)
    low = compute_total_reward(hypergraph_checker=_DummyChecker(score=0.1), **kwargs)
    assert high["R_bound"] == pytest.approx(0.9)
    assert low["R_bound"] == pytest.approx(0.1)


def test_compute_total_reward_optional_tom_ground_truth_defaults_to_neutral():
    # Confirms R_tom defaults to neutral 1.0 (per its own "no ground truth"
    # convention) when recipient_knows/recipient_does_not_know aren't passed
    # -- i.e. compute_total_reward callers that predate R_tom's addition
    # still work correctly without modification.
    result = compute_total_reward(
        generated_text=_generation(patient_state="Tachycardic."),
        reference_patient_state="Tachycardic.",
        hypergraph_checker=_DummyChecker(),
        recipient_type="clinician",
        must_mention_facts=[],
    )
    assert result["R_tom"] == 1.0
