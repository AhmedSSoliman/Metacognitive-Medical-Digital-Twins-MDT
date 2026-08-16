"""
core/rewards/forecast.py -- R_forecast: quantitative forecast accuracy.

Ported verbatim from the source repo's training/rewards.py.
"""

from __future__ import annotations

import numpy as np

from core.parsing import ParsedStreams


# ---------------------------------------------------------------------------
# R_forecast: quantitative forecast accuracy (structured <forecast> stream)
# ---------------------------------------------------------------------------

def reward_forecast_accuracy(parsed: ParsedStreams, true_future_values: dict[str, float],
                              tolerance_fraction: float = 0.15) -> float:
    """Scores the structured <forecast> stream against ground-truth future
    values (e.g. actual MAP/lactate at the +6h horizon, computed the same
    way data/preprocessing.py's deterioration labels are computed).

    `true_future_values` maps variable name (matching the forecast stream's
    keys, e.g. "MAP_6h") to the true observed value. Score per variable is
    1.0 if the true value falls within the predicted [low, high] interval,
    decaying based on how far outside the interval it falls otherwise (not
    a hard 0/1 cliff, so the reward gives a useful gradient even for
    near-miss predictions rather than treating "just outside the interval"
    the same as "wildly wrong").

    If the model correctly used "not applicable" (empty forecast_values) AND
    no true future values were expected for this prompt, this returns a
    neutral 1.0 -- opting out correctly is not penalized. If the model
    forecasts nothing when a real forecast WAS expected, this returns 0.0.
    """
    if not true_future_values:
        return 1.0 if not parsed.forecast_values else 0.5  # forecasting when none was expected: mild penalty, not zero
    if not parsed.forecast_values:
        return 0.0  # a forecast was expected but the model opted out or produced nothing parseable

    scores = []
    for var, true_val in true_future_values.items():
        if var not in parsed.forecast_values:
            scores.append(0.0)
            continue
        entry = parsed.forecast_values[var]
        if entry.low <= true_val <= entry.high:
            scores.append(1.0)
        else:
            # Distance-based decay outside the interval, normalized by a
            # tolerance band around the interval width so a near-miss scores
            # meaningfully higher than a wildly-off prediction.
            interval_width = max(entry.high - entry.low, 1e-6)
            if true_val < entry.low:
                miss_distance = entry.low - true_val
            else:
                miss_distance = true_val - entry.high
            decay = max(0.0, 1.0 - (miss_distance / (tolerance_fraction * interval_width + interval_width)))
            scores.append(decay)
    return float(np.mean(scores))
