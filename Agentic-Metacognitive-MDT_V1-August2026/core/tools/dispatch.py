"""
core/tools/dispatch.py

Ported from the source repo's agents/tool_use.py
(../Agentic-DT_V1-July/agents/tool_use.py). Only the two intra-project
imports were rewritten for the new layout:
  - `from models.multi_stream import parse_streams` -> `from core.parsing
    import parse_streams` (importing the torch-free parser directly instead
    of via the torch-importing model wrapper, which also makes this whole
    module importable without torch installed).
  - `from hypergraph.verification import InterimRuleBasedChecker` (in the
    __main__ smoke test) -> `from core.hypergraph.verification import ...`.
References below to "agents/rollout_service.py" now correspond to
training/rollout.py, and "data/build_grpo_prompts.py" to
core/cohort/grpo_prompts.py.

Phase 4: ReAct-style agentic tool use.

Lets the model, mid-reasoning inside <think>, issue a structured tool call
(as a JSON object) to query the hypergraph or retrieve additional patient
data, then resumes generation with the tool's result appended to context.

Tool-call syntax expected inside <think>:
    {"tool": "query_hypergraph", "args": {"claimed_abnormalities": ["tachycardia", "hypotension"]}}
    {"tool": "get_recent_labs", "args": {"variable": "lactate", "hours": 6}}

This module implements the tool EXECUTION side. It does not change the
Multi-Stream generation format -- tool calls are just structured text that
appears inside <think>, parsed here and re-injected as an observation.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

TOOL_CALL_RE = re.compile(r'\{[^{}]*"tool"\s*:\s*"([a-zA-Z_]+)"[^{}]*\}')


@dataclass
class ToolCallResult:
    tool_name: str
    success: bool
    result: Optional[dict]
    latency_ms: float
    error: Optional[str] = None


class ToolRegistry:
    """Maps tool names to callables. Each tool function takes a dict of args
    and returns a JSON-serializable dict result.
    """

    def __init__(self):
        self._tools: dict[str, Callable[[dict], dict]] = {}

    def register(self, name: str, fn: Callable[[dict], dict]):
        self._tools[name] = fn

    def call(self, tool_name: str, args: dict, timeout_s: float = 5.0) -> ToolCallResult:
        if tool_name not in self._tools:
            return ToolCallResult(tool_name, False, None, 0.0, error=f"Unknown tool: {tool_name}")

        start = time.time()
        try:
            result = self._tools[tool_name](args)
            latency_ms = (time.time() - start) * 1000
            return ToolCallResult(tool_name, True, result, latency_ms)
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.warning("Tool call %s failed: %s", tool_name, e)
            return ToolCallResult(tool_name, False, None, latency_ms, error=str(e))


def extract_tool_calls(think_text: str) -> list[dict]:
    """Finds JSON-looking tool-call objects inside a <think> block.

    Uses bracket-depth counting rather than a flat regex, since tool calls
    have nested objects (e.g. {"tool": ..., "args": {...}}) that a
    single-level regex cannot match correctly.
    """
    calls = []
    start_marker = re.compile(r'\{\s*"tool"\s*:')
    for start_match in start_marker.finditer(think_text):
        start = start_match.start()
        depth = 0
        end = None
        for i in range(start, len(think_text)):
            ch = think_text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            continue  # unbalanced braces, skip
        candidate = think_text[start:end]
        try:
            calls.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return calls


# ---------------------------------------------------------------------------
# Default tool implementations
# ---------------------------------------------------------------------------

def make_default_registry(hypergraph_checker, patient_timeseries=None,
                           prediction_time_cutoff: Optional[str] = None) -> ToolRegistry:
    """`patient_timeseries`: a DataFrame of THIS rollout's specific patient's
    [charttime, variable, value] readings (a slice of the combined
    vitals+labs table hypergraph/construction.py's Phase 3 pipeline builds
    -- see agents/rollout_service.py's rollout_worker for where this slice
    is cut per-request), or None if no per-patient data context is
    available (get_recent_labs then falls back to a clear "not configured"
    response rather than fabricating an answer).

    `prediction_time_cutoff`: an ISO-format timestamp string marking the
    point this rollout's prompt was generated as of (data/build_grpo_prompts.py's
    prediction_time column). get_recent_labs only ever returns readings
    STRICTLY BEFORE this cutoff -- without it, a "recent labs" tool call
    would have no leakage-safe boundary and could return values from AFTER
    the point the prompt itself was constructed from, silently giving the
    model information it wasn't supposed to have access to yet.
    """
    registry = ToolRegistry()

    def query_hypergraph(args: dict) -> dict:
        claimed = args.get("claimed_abnormalities", [])
        # Reuse the checker's flag-detection machinery by feeding it a
        # synthetic sentence built from the claimed terms.
        synthetic_text = " and ".join(claimed)
        score = hypergraph_checker.check(synthetic_text)
        return {
            "claimed_abnormalities": claimed,
            "physiologically_grounded_score": score,
            "interpretation": "consistent with known patterns" if score > 0.5 else "not supported by derived hypergraph",
        }

    def get_recent_labs(args: dict) -> dict:
        if patient_timeseries is None or prediction_time_cutoff is None:
            return {"error": "No per-patient data context configured for get_recent_labs in this "
                              "rollout context (requires both patient_timeseries and "
                              "prediction_time_cutoff -- see agents/rollout_service.py's "
                              "rollout_worker for how these get wired in per-request)."}
        variable = args.get("variable")
        hours = args.get("hours", 6)
        if not variable:
            return {"error": "get_recent_labs requires an 'variable' argument."}

        import pandas as pd
        cutoff = pd.Timestamp(prediction_time_cutoff)
        window_start = cutoff - pd.Timedelta(hours=hours)
        # Strictly BEFORE cutoff (not <=) -- prediction_time is the moment
        # this vignette's prompt was generated from, so a reading recorded
        # exactly AT that moment is still information the model would not
        # plausibly have had a chance to react to yet.
        window = patient_timeseries[
            (patient_timeseries["variable"] == variable)
            & (patient_timeseries["charttime"] >= window_start)
            & (patient_timeseries["charttime"] < cutoff)
        ].sort_values("charttime")

        if window.empty:
            return {"variable": variable, "hours": hours, "readings": [],
                     "note": f"No {variable} readings found in the {hours}h window before prediction_time."}

        readings = [{"charttime": row["charttime"].isoformat(), "value": float(row["value"])}
                    for _, row in window.iterrows()]
        return {
            "variable": variable,
            "hours": hours,
            "readings": readings,
            "most_recent_value": readings[-1]["value"],
            "trend": "rising" if len(readings) >= 2 and readings[-1]["value"] > readings[0]["value"]
                     else ("falling" if len(readings) >= 2 and readings[-1]["value"] < readings[0]["value"]
                           else "insufficient_data_for_trend"),
        }

    registry.register("query_hypergraph", query_hypergraph)
    registry.register("get_recent_labs", get_recent_labs)
    return registry


def run_agentic_turn(model_generate_fn: Callable[[str], str], prompt: str,
                      registry: ToolRegistry, max_tool_calls: int = 3) -> dict:
    """Runs one ReAct-style loop: generate -> extract tool calls from <think>
    -> execute -> append observations -> regenerate, up to max_tool_calls times
    or until no new tool call appears.

    `model_generate_fn` should be a closure over MultiStreamModel.generate
    that returns a single string (the first generated sequence).
    """
    from core.parsing import parse_streams

    tool_call_log = []
    current_prompt = prompt

    for turn in range(max_tool_calls):
        generated = model_generate_fn(current_prompt)
        parsed = parse_streams(generated)
        if parsed.think is None:
            break

        calls = extract_tool_calls(parsed.think)
        if not calls:
            return {"final_generation": generated, "tool_calls": tool_call_log, "turns_used": turn}

        observations = []
        for call in calls:
            tool_name = call.get("tool")
            args = call.get("args", {})
            result = registry.call(tool_name, args)
            tool_call_log.append(result)
            observations.append(
                f"[Tool result for {tool_name}: "
                f"{'SUCCESS' if result.success else 'FAILED - ' + str(result.error)} "
                f"{json.dumps(result.result) if result.result else ''}]"
            )

        current_prompt = (
            f"{prompt}\n\n### Previous reasoning and tool results\n"
            f"{parsed.think}\n" + "\n".join(observations) +
            "\n\nContinue your reasoning and provide the final structured response."
        )

    generated = model_generate_fn(current_prompt)
    return {"final_generation": generated, "tool_calls": tool_call_log, "turns_used": max_tool_calls}


if __name__ == "__main__":
    from core.hypergraph.verification import InterimRuleBasedChecker
    checker = InterimRuleBasedChecker()
    registry = make_default_registry(checker)

    think_block = (
        'Let me check this combination. '
        '{"tool": "query_hypergraph", "args": {"claimed_abnormalities": ["tachycardia", "hypotension"]}} '
        'That looks consistent with shock physiology.'
    )
    calls = extract_tool_calls(think_block)
    print("Extracted calls:", calls)
    for call in calls:
        result = registry.call(call["tool"], call["args"])
        print(result)
