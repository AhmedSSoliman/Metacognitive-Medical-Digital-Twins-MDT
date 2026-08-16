"""
tests/test_rewards/test_reward_logic.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_reward_logic.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Tests the pure-logic reward components (format compliance, physiological
grounding, empathy framing, tool-call JSON validity, forecast accuracy) --
none of these need the sentence-embedding encoder, and after a fix applied
during development (moving the `sentence_transformers` import to be lazy,
inside get_sentence_encoder(), rather than at module level), importing them
no longer requires torch to be installed at all.

The three reward functions that DO need embedding similarity (semantic
fidelity, context retention, metacognitive self-correction) are tested
separately in tests/test_reward_semantic.py, marked requires_torch, since
they call get_sentence_encoder() internally.
"""

import sys
from pathlib import Path

from core.rewards import (
    reward_format, reward_physio_grounding, reward_empathy,
    reward_tool_call, reward_forecast_accuracy,
)
from core.parsing import parse_streams, ForecastEntry


# ---------------------------------------------------------------------------
# R_format
# ---------------------------------------------------------------------------

def test_reward_format_well_formed_scores_perfect(well_formed_generation):
    assert reward_format(well_formed_generation) == 1.0


def test_reward_format_wrong_order_scores_partial():
    bad_order = (
        "<patient_state>b</patient_state><think>a</think>"
        "<forecast>not applicable</forecast><user_belief>c</user_belief>"
    )
    score = reward_format(bad_order)
    assert 0.0 < score < 1.0


def test_reward_format_partial_tags_scores_lower_than_full():
    partial = "<think>a</think><patient_state>b</patient_state>"
    full = "<think>a</think><patient_state>b</patient_state><forecast>not applicable</forecast><user_belief>c</user_belief>"
    assert reward_format(partial) < reward_format(full)


# ---------------------------------------------------------------------------
# R_physio
# ---------------------------------------------------------------------------

def test_physio_grounding_rewards_clinical_terminology():
    grounded_text = (
        "<think>x</think><patient_state>Tachycardic, hypotensive, with hyperlactatemia "
        "concerning for shock.</patient_state><forecast>not applicable</forecast>"
        "<user_belief>clinician</user_belief>"
    )
    ungrounded_text = (
        "<think>x</think><patient_state>The patient seems okay I guess.</patient_state>"
        "<forecast>not applicable</forecast><user_belief>clinician</user_belief>"
    )
    grounded_score = reward_physio_grounding(parse_streams(grounded_text))
    ungrounded_score = reward_physio_grounding(parse_streams(ungrounded_text))
    assert grounded_score > ungrounded_score


# ---------------------------------------------------------------------------
# R_emp
# ---------------------------------------------------------------------------

def test_empathy_rewards_plain_language_for_non_clinician():
    plain_language = (
        "<think>x</think><patient_state>y</patient_state><forecast>not applicable</forecast>"
        "<user_belief>Your blood pressure is a bit low and your heart rate is fast, "
        "which we're keeping an eye on.</user_belief>"
    )
    jargon = (
        "<think>x</think><patient_state>y</patient_state><forecast>not applicable</forecast>"
        "<user_belief>Hypotension and tachycardia noted.</user_belief>"
    )
    plain_score = reward_empathy(parse_streams(plain_language), "patient")
    jargon_score = reward_empathy(parse_streams(jargon), "patient")
    assert plain_score > jargon_score


# ---------------------------------------------------------------------------
# R_tool
# ---------------------------------------------------------------------------

def test_tool_call_reward_neutral_when_no_tool_call_attempted():
    text = "<think>Just reasoning, no tools needed here.</think>"
    assert reward_tool_call(text) == 1.0


def test_tool_call_reward_handles_nested_json_correctly():
    # Regression test for a real bug: the original regex
    # r'\{[^{}]*"tool"[^{}]*\}' could not match tool calls with a nested
    # "args" object (e.g. {"tool": "x", "args": {"key": "val"}}), since
    # [^{}]* forbids any nested braces. Fixed with bracket-depth counting.
    text = (
        '<think>Checking. {"tool": "query_hypergraph", "args": '
        '{"claimed_abnormalities": ["tachycardia"]}} done.</think>'
    )
    score = reward_tool_call(text)
    assert score == 1.0, "Well-formed nested-JSON tool call should score as fully valid"


def test_tool_call_reward_penalizes_malformed_json():
    text = '<think>{"tool": "broken", "args": {not valid json}}</think>'
    score = reward_tool_call(text)
    assert score == 0.0


# ---------------------------------------------------------------------------
# R_forecast
# ---------------------------------------------------------------------------

def test_forecast_reward_perfect_when_true_value_inside_interval():
    parsed = parse_streams(
        "<think>x</think><patient_state>y</patient_state>"
        "<forecast>MAP_6h: 58 [52-64]</forecast><user_belief>clinician</user_belief>"
    )
    score = reward_forecast_accuracy(parsed, {"MAP_6h": 60})
    assert score == 1.0


def test_forecast_reward_decays_for_near_miss():
    parsed = parse_streams(
        "<think>x</think><patient_state>y</patient_state>"
        "<forecast>MAP_6h: 58 [52-64]</forecast><user_belief>clinician</user_belief>"
    )
    score = reward_forecast_accuracy(parsed, {"MAP_6h": 65})  # just outside [52, 64]
    assert 0.5 < score < 1.0


def test_forecast_reward_near_zero_for_wildly_off_prediction():
    parsed = parse_streams(
        "<think>x</think><patient_state>y</patient_state>"
        "<forecast>MAP_6h: 58 [52-64]</forecast><user_belief>clinician</user_belief>"
    )
    score = reward_forecast_accuracy(parsed, {"MAP_6h": 120})
    assert score < 0.1


def test_forecast_reward_correct_opt_out_scores_neutral():
    parsed = parse_streams(
        "<think>x</think><patient_state>y</patient_state>"
        "<forecast>not applicable</forecast><user_belief>clinician</user_belief>"
    )
    score = reward_forecast_accuracy(parsed, {})  # nothing was expected, model correctly opted out
    assert score == 1.0


def test_forecast_reward_penalizes_missing_forecast_when_expected():
    parsed = parse_streams(
        "<think>x</think><patient_state>y</patient_state>"
        "<forecast>not applicable</forecast><user_belief>clinician</user_belief>"
    )
    score = reward_forecast_accuracy(parsed, {"MAP_6h": 60})  # a forecast WAS expected
    assert score == 0.0
