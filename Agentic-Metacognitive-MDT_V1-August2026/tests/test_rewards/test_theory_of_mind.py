"""
tests/test_rewards/test_theory_of_mind.py

Tests core/rewards/theory_of_mind.py::reward_theory_of_mind. Most cases use
exact substring matches (no encoder call needed, matching
test_reward_logic.py's no-torch pattern); one case forces the embedding
similarity fallback and is marked requires_torch.
"""

import pytest

from core.parsing import parse_streams
from core.rewards.theory_of_mind import reward_theory_of_mind


def _generation_with_belief(user_belief: str) -> str:
    return (
        f"<think>x</think><patient_state>y</patient_state>"
        f"<forecast>not applicable</forecast><user_belief>{user_belief}</user_belief>"
    )


def test_neutral_score_when_no_ground_truth_available():
    parsed = parse_streams(_generation_with_belief("anything at all"))
    assert reward_theory_of_mind(parsed, None, None) == 1.0
    assert reward_theory_of_mind(parsed, [], []) == 1.0


def test_zero_when_user_belief_missing_but_ground_truth_present():
    parsed = parse_streams("<think>x</think><patient_state>y</patient_state><forecast>not applicable</forecast>")
    assert reward_theory_of_mind(parsed, ["the diagnosis"], []) == 0.0


def test_rewards_not_restating_already_known_fact():
    # Recipient already knows the diagnosis -- NOT restating it scores higher
    # than restating it as if new.
    restates = _generation_with_belief(
        "Addressed to the family; the diagnosis is DKA, which is a serious condition."
    )
    omits = _generation_with_belief(
        "Addressed to the family; focusing on what to expect over the next few hours."
    )
    parsed_restates = parse_streams(restates)
    parsed_omits = parse_streams(omits)

    score_restates = reward_theory_of_mind(parsed_restates, ["the diagnosis"], [])
    score_omits = reward_theory_of_mind(parsed_omits, ["the diagnosis"], [])
    assert score_omits > score_restates


def test_rewards_addressing_unknown_fact_via_exact_substring():
    addresses = _generation_with_belief("Explaining the diagnosis clearly since it hasn't been discussed yet.")
    silent = _generation_with_belief("Keeping the tone reassuring and brief.")

    parsed_addresses = parse_streams(addresses)
    parsed_silent = parse_streams(silent)

    score_addresses = reward_theory_of_mind(parsed_addresses, [], ["the diagnosis"])
    score_silent = reward_theory_of_mind(parsed_silent, [], ["the diagnosis"])
    assert score_addresses == 1.0
    assert score_silent == 0.0
    assert score_addresses > score_silent


def test_averages_across_multiple_facts():
    # Addresses one unknown fact but not the other -- expect a mid-range score.
    parsed = parse_streams(_generation_with_belief(
        "Explaining the diagnosis clearly since it hasn't been discussed yet."
    ))
    score = reward_theory_of_mind(parsed, [], ["the diagnosis", "the raw numeric trend values"])
    assert 0.0 < score < 1.0


@pytest.mark.requires_torch
def test_embedding_fallback_catches_paraphrased_mention():
    # No exact substring match against "the diagnosis or working clinical
    # assessment" (missing the leading "the diagnosis or"), so this only
    # scores as "addressed" via the embedding-similarity fallback --
    # verified against the real encoder: cos_sim ~0.60 for this phrasing,
    # ~0.46 for a more tangential one (see this file's git history / the
    # commit that fixed this test for the measured values that motivated
    # the 0.55 default threshold).
    parsed = parse_streams(_generation_with_belief(
        "This explains her diagnosis and working clinical assessment directly."
    ))
    score = reward_theory_of_mind(parsed, [], ["the diagnosis or working clinical assessment"])
    assert score == 1.0
