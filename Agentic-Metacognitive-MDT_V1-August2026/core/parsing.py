"""
core/parsing.py

Four-stream parser and tag validation -- the parsing-logic half of the source
repo's models/stream_parsing.py (../Agentic-DT_V1-July/models/stream_parsing.py).
The schema/spec constants (STREAM_TAGS, STREAM_SYSTEM_PROMPT,
FORECAST_FORMAT_EXAMPLE) now live in core/schema.py and are imported (and
re-exported) here, so every existing `from ... import parse_streams,
STREAM_TAGS` style call site keeps working against one module.

Like core/schema.py, this file is deliberately FREE of any
torch/transformers/peft dependency -- see core/schema.py's docstring for the
original bug that motivated that constraint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Re-exported so callers can do `from core.parsing import STREAM_TAGS` exactly
# as they previously did `from models.stream_parsing import STREAM_TAGS`.
from core.schema import (  # noqa: F401
    STREAM_TAGS,
    FORECAST_FORMAT_EXAMPLE,
    STREAM_SYSTEM_PROMPT,
)

_STREAM_RE = {
    tag: re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL) for tag in STREAM_TAGS
}

# Parses lines like "MAP_6h: 58 [52-64]" or "lactate_6h: 3.1 [2.4-3.9]".
# Verified during development to correctly handle: extra internal whitespace,
# negative values with spaces around the range separator, and negative
# values WITHOUT spaces around the separator (a genuinely ambiguous case,
# e.g. "-5.1--1.0", which this pattern correctly reads as two numbers, not one).
_FORECAST_LINE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(-?\d+\.?\d*)\s*\[\s*(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)\s*\]",
    re.MULTILINE,
)


@dataclass
class ForecastEntry:
    variable: str
    value: float
    low: float
    high: float


def parse_forecast_text(forecast_text: Optional[str]) -> dict[str, ForecastEntry]:
    """Parses the structured forecast sub-format into {variable: ForecastEntry}.
    Returns an empty dict if forecast_text is None, empty, or says
    "not applicable" (the model's allowed way to opt out of forecasting for
    a given prompt, per STREAM_SYSTEM_PROMPT).
    """
    if not forecast_text or forecast_text.strip().lower() == "not applicable":
        return {}
    entries = {}
    for m in _FORECAST_LINE_RE.finditer(forecast_text):
        var, value, low, high = m.groups()
        try:
            entries[var] = ForecastEntry(variable=var, value=float(value), low=float(low), high=float(high))
        except ValueError:
            continue  # malformed number in an otherwise-matching line -- skip rather than crash
    return entries


@dataclass
class ParsedStreams:
    think: Optional[str]
    patient_state: Optional[str]
    forecast: Optional[str]                    # raw forecast stream text
    forecast_values: dict[str, ForecastEntry]   # parsed structured forecast entries
    user_belief: Optional[str]
    well_formed: bool  # True iff all four streams present, in order, no extra text outside tags


def parse_streams(generated_text: str) -> ParsedStreams:
    """Extracts the four streams from a raw generation. Used both for
    reward computation (Phase 2) and for downstream hypergraph verification
    (Phase 3, which only ever looks at the <patient_state> stream)."""
    matches = {}
    for tag in STREAM_TAGS:
        m = _STREAM_RE[tag].search(generated_text)
        matches[tag] = m.group(1).strip() if m else None

    # Well-formedness: all four present AND in the expected order AND no
    # significant stray text outside the tags (a loose check -- exact
    # compliance scoring for R_format lives in training/rewards.py).
    order_ok = False
    positions = []
    for tag in STREAM_TAGS:
        idx = generated_text.find(f"<{tag}>")
        positions.append(idx)
    if all(p != -1 for p in positions) and positions == sorted(positions):
        order_ok = True

    well_formed = all(v is not None for v in matches.values()) and order_ok

    return ParsedStreams(
        think=matches["think"],
        patient_state=matches["patient_state"],
        forecast=matches["forecast"],
        forecast_values=parse_forecast_text(matches["forecast"]),
        user_belief=matches["user_belief"],
        well_formed=well_formed,
    )


if __name__ == "__main__":
    # Smoke test of the parser only (no model load, no torch import needed --
    # this is the whole point of this module's existence).
    example = (
        "<think>HR trending up, MAP trending down -- possible early shock.</think>"
        "<patient_state>Tachycardic, borderline hypotensive, lactate pending.</patient_state>"
        "<forecast>MAP_6h: 58 [52-64]\nlactate_6h: 3.1 [2.4-3.9]</forecast>"
        "<user_belief>Reader is likely a bedside nurse; keep it concrete and action-oriented.</user_belief>"
    )
    parsed = parse_streams(example)
    print(parsed)
    assert parsed.well_formed
    assert "MAP_6h" in parsed.forecast_values
    assert parsed.forecast_values["MAP_6h"].value == 58.0

    example_na = (
        "<think>General question, no forecast needed.</think>"
        "<patient_state>N/A</patient_state>"
        "<forecast>not applicable</forecast>"
        "<user_belief>Clinician.</user_belief>"
    )
    parsed_na = parse_streams(example_na)
    assert parsed_na.well_formed
    assert parsed_na.forecast_values == {}
    print("Torch-free smoke test passed.")
