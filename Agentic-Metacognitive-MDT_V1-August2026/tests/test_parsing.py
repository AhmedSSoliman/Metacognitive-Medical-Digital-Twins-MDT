"""
tests/test_parsing.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_stream_parsing.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Formalizes the manual parser verification done during development into
reusable tests: well-formedness detection, forecast sub-format parsing
(including edge cases: negative numbers, no-space ranges, "not applicable"
opt-out), and the tag-order check.

These do NOT require torch/a loaded model -- parse_streams and
parse_forecast_text are pure regex/string logic, importable in isolation.
"""

import sys
from pathlib import Path

from core.parsing import parse_streams, parse_forecast_text, STREAM_TAGS


def test_well_formed_generation_parses_correctly(well_formed_generation):
    parsed = parse_streams(well_formed_generation)
    assert parsed.well_formed
    assert parsed.think is not None
    assert parsed.patient_state is not None
    assert parsed.forecast is not None
    assert parsed.user_belief is not None


def test_missing_stream_is_not_well_formed():
    # Missing <forecast> entirely -- this is the exact bug class caught and
    # fixed in evaluation/report.py's own smoke test fixture during
    # development (a fixture that predated the forecast stream being added).
    text = (
        "<think>ok</think>"
        "<patient_state>fine</patient_state>"
        "<user_belief>clinician</user_belief>"
    )
    parsed = parse_streams(text)
    assert not parsed.well_formed
    assert parsed.forecast is None


def test_wrong_tag_order_is_not_well_formed():
    text = (
        "<patient_state>fine</patient_state>"
        "<think>ok</think>"
        "<forecast>not applicable</forecast>"
        "<user_belief>clinician</user_belief>"
    )
    parsed = parse_streams(text)
    assert not parsed.well_formed


def test_stream_tags_order_matches_system_prompt_convention():
    # Regression test for a real bug caught during development: STREAM_TAGS
    # was briefly defined in a different order than the system prompt text
    # described, which would have made every well-formed generation fail
    # the order check. Pinning the expected order here catches that class
    # of mistake immediately if it's ever reintroduced.
    assert STREAM_TAGS == ["think", "patient_state", "forecast", "user_belief"]


def test_forecast_parsing_basic():
    values = parse_forecast_text("MAP_6h: 58 [52-64]\nlactate_6h: 3.1 [2.4-3.9]")
    assert values["MAP_6h"].value == 58.0
    assert values["MAP_6h"].low == 52.0
    assert values["MAP_6h"].high == 64.0
    assert values["lactate_6h"].value == 3.1


def test_forecast_parsing_not_applicable_opts_out_cleanly():
    assert parse_forecast_text("not applicable") == {}
    assert parse_forecast_text(None) == {}
    assert parse_forecast_text("") == {}


def test_forecast_parsing_handles_extra_whitespace():
    values = parse_forecast_text("MAP_6h: 58 [ 52 - 64 ]")
    assert values["MAP_6h"].value == 58.0


def test_forecast_parsing_handles_negative_values_with_spaces():
    values = parse_forecast_text("base_excess: -3.2 [-8.1 - -2.3]")
    assert values["base_excess"].low == -8.1
    assert values["base_excess"].high == -2.3


def test_forecast_parsing_handles_negative_values_without_spaces():
    # The genuinely ambiguous case: is "-5.1--1.0" one number or two? Verified
    # during development that the regex correctly reads this as two separate
    # negative numbers (-5.1 and -1.0), not one malformed token.
    values = parse_forecast_text("base_excess_6h: -3.2 [-5.1--1.0]")
    assert values["base_excess_6h"].low == -5.1
    assert values["base_excess_6h"].high == -1.0


def test_forecast_parsing_ignores_garbage_text():
    assert parse_forecast_text("garbage text with no valid lines") == {}
