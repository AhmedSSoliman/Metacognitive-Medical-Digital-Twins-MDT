"""
core/rewards/format.py -- R_format: XML structural tag compliance.

Ported verbatim from the source repo's training/rewards.py
(../Agentic-DT_V1-July/training/rewards.py). Only the import of
parse_streams/STREAM_TAGS was rewritten to the new core.parsing location.
Pure regex/string logic -- no torch, no numpy.
"""

from __future__ import annotations

from core.parsing import parse_streams, STREAM_TAGS


# ---------------------------------------------------------------------------
# R_format: XML structural tag compliance
# ---------------------------------------------------------------------------

def reward_format(generated_text: str) -> float:
    parsed = parse_streams(generated_text)
    if parsed.well_formed:
        return 1.0
    # Partial credit: how many of the three tags are present at all, regardless of order
    present = sum(1 for tag in STREAM_TAGS if f"<{tag}>" in generated_text and f"</{tag}>" in generated_text)
    return 0.3 * (present / len(STREAM_TAGS))
