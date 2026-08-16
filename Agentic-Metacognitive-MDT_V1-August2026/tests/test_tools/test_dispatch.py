"""
tests/test_tools/test_dispatch.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_tool_use.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Tests agents.tool_use's extract_tool_calls -- specifically a real bug fix:
the original implementation used a flat regex, r'\\{[^{}]*"tool"[^{}]*\\}',
which cannot match tool calls containing a nested "args" object (since
[^{}]* forbids any nested braces at all). Fixed with bracket-depth counting.
No torch dependency -- this is pure regex/JSON parsing logic.
"""

import sys
from pathlib import Path

import pandas as pd

from core.tools.dispatch import extract_tool_calls, ToolRegistry, ToolCallResult, make_default_registry
from core.hypergraph.verification import InterimRuleBasedChecker


def test_extract_simple_tool_call_no_nesting():
    text = 'Checking. {"tool": "get_recent_labs", "args": {}} done.'
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "get_recent_labs"


def test_extract_tool_call_with_nested_args_object():
    """Regression test for the real bug: a flat single-level regex could not
    match this nested structure at all, silently returning zero tool calls
    even though a well-formed one was present."""
    text = (
        'Let me check this. {"tool": "query_hypergraph", "args": '
        '{"claimed_abnormalities": ["tachycardia", "hypotension"]}} '
        'That looks consistent with shock physiology.'
    )
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["tool"] == "query_hypergraph"
    assert calls[0]["args"]["claimed_abnormalities"] == ["tachycardia", "hypotension"]


def test_extract_multiple_tool_calls():
    text = (
        '{"tool": "query_hypergraph", "args": {"x": 1}} and then '
        '{"tool": "get_recent_labs", "args": {"variable": "lactate"}}'
    )
    calls = extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["tool"] == "query_hypergraph"
    assert calls[1]["tool"] == "get_recent_labs"


def test_extract_ignores_malformed_json():
    text = '{"tool": "broken", "args": {not valid json here}}'
    calls = extract_tool_calls(text)
    assert calls == []


def test_extract_returns_empty_list_when_no_tool_call_present():
    text = "Just reasoning about the case, no tool calls needed here."
    assert extract_tool_calls(text) == []


def test_tool_registry_calls_registered_function():
    registry = ToolRegistry()
    registry.register("echo", lambda args: {"echoed": args.get("value")})
    result = registry.call("echo", {"value": 42})
    assert isinstance(result, ToolCallResult)
    assert result.success
    assert result.result["echoed"] == 42


def test_tool_registry_reports_failure_for_unknown_tool():
    registry = ToolRegistry()
    result = registry.call("nonexistent_tool", {})
    assert not result.success
    assert "Unknown tool" in result.error


def test_tool_registry_catches_exceptions_from_tool_function():
    registry = ToolRegistry()
    def broken_tool(args):
        raise ValueError("something went wrong inside the tool")
    registry.register("broken", broken_tool)
    result = registry.call("broken", {})
    assert not result.success
    assert "something went wrong" in result.error


# ---------------------------------------------------------------------------
# get_recent_labs (training/rollout.py wires this to a real per-patient
# timeseries slice + prediction_time cutoff; these tests exercise the
# underlying logic directly via make_default_registry, without a real
# rollout worker/model).
# ---------------------------------------------------------------------------

def _sample_timeseries():
    return pd.DataFrame({
        "charttime": pd.to_datetime([
            "2026-01-01T00:00:00", "2026-01-01T02:00:00", "2026-01-01T04:00:00",
            "2026-01-01T06:00:00",  # exactly AT the cutoff below -- must be excluded
            "2026-01-01T08:00:00",  # AFTER the cutoff -- must be excluded
        ]),
        "variable": ["lactate", "lactate", "lactate", "lactate", "lactate"],
        "value": [1.4, 1.8, 2.6, 3.5, 5.0],
    })


def test_get_recent_labs_without_context_returns_not_configured_error():
    """No patient_timeseries/prediction_time_cutoff provided (the state
    every call was in before this fix) -- must fail clearly, not silently
    fabricate an answer."""
    registry = make_default_registry(InterimRuleBasedChecker())
    result = registry.call("get_recent_labs", {"variable": "lactate", "hours": 6})
    assert result.success  # the tool function itself doesn't raise
    assert "error" in result.result
    assert "not configured" in result.result["error"].lower() or "No per-patient" in result.result["error"]


def test_get_recent_labs_returns_readings_within_window():
    registry = make_default_registry(
        InterimRuleBasedChecker(),
        patient_timeseries=_sample_timeseries(),
        prediction_time_cutoff="2026-01-01T06:00:00",
    )
    result = registry.call("get_recent_labs", {"variable": "lactate", "hours": 6})
    assert result.success
    values = [r["value"] for r in result.result["readings"]]
    # The 06:00 (== cutoff) and 08:00 (after cutoff) readings must NOT appear --
    # this is the leakage-safety guarantee the cutoff exists to provide.
    assert values == [1.4, 1.8, 2.6]
    assert result.result["most_recent_value"] == 2.6
    assert result.result["trend"] == "rising"


def test_get_recent_labs_excludes_readings_at_or_after_cutoff():
    """Direct regression test for the leakage-safety guarantee: a reading
    recorded exactly AT prediction_time must be excluded, not just ones
    strictly after it."""
    registry = make_default_registry(
        InterimRuleBasedChecker(),
        patient_timeseries=_sample_timeseries(),
        prediction_time_cutoff="2026-01-01T04:00:00",  # a reading exists at exactly this time
    )
    result = registry.call("get_recent_labs", {"variable": "lactate", "hours": 6})
    values = [r["value"] for r in result.result["readings"]]
    assert 2.6 not in values  # the 04:00 reading (== cutoff) must be excluded
    assert values == [1.4, 1.8]


def test_get_recent_labs_empty_window_returns_empty_readings_not_error():
    registry = make_default_registry(
        InterimRuleBasedChecker(),
        patient_timeseries=_sample_timeseries(),
        prediction_time_cutoff="2026-01-01T06:00:00",
    )
    result = registry.call("get_recent_labs", {"variable": "creatinine", "hours": 6})
    assert result.success
    assert result.result["readings"] == []
