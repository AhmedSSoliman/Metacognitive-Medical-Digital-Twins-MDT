"""
core/rewards/tool_use.py -- R_tool: agentic tool-call verification.

Ported verbatim from the source repo's training/rewards.py. Note this is the
REWARD side (a syntax-validity proxy); the tool EXECUTION side lives in
core/tools/dispatch.py.
"""

from __future__ import annotations

import re
from typing import Optional

from core.parsing import parse_streams


# ---------------------------------------------------------------------------
# R_tool: agentic tool-call verification (Phase 4 relevant, stubbed here for Phase 2)
# ---------------------------------------------------------------------------

def reward_tool_call(generated_text: str, expected_tool_schema: Optional[dict] = None) -> float:
    """In Phase 2, before Phase 4's tool-use is wired in, this simply checks
    whether any tool-call-like syntax appearing in <think> is well-formed
    JSON (a weak proxy). Phase 4's rollout service should replace this with
    real tool-execution success, not a syntax check.
    """
    parsed = parse_streams(generated_text)
    if parsed.think is None:
        return 0.0

    import json as _json
    # Bracket-depth matching (not a flat regex) since tool calls contain
    # nested "args" objects that a single-level regex cannot capture.
    text = parsed.think
    start_marker = re.compile(r'\{\s*"tool"\s*:')
    tool_call_matches = []
    for start_match in start_marker.finditer(text):
        start = start_match.start()
        depth = 0
        end = None
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is not None:
            tool_call_matches.append(text[start:end])

    if not tool_call_matches:
        return 1.0  # no tool call attempted -- neutral, not penalized in Phase 2
    valid = 0
    for m in tool_call_matches:
        try:
            _json.loads(m)
            valid += 1
        except _json.JSONDecodeError:
            pass
    return valid / len(tool_call_matches)
