"""
evaluation/retention.py -- Endpoint 3: Trajectory Accuracy (MAE over
forecasted numeric values).

Ported verbatim from the source repo's evaluation/metrics.py
(evaluate_trajectory_accuracy).

NAMING NOTE, worth reading before using this module: the target tree
describes this file as "Context retention MAE over multi-day windows". No
such metric exists in the source repo. What DOES exist, and what is ported
here, is `evaluate_trajectory_accuracy` -- an MAE between FORECASTED and
observed future values (Endpoint 3, e.g. predicted vs. true MAP at +6h). It
is an MAE, but over forecast trajectories, not over context retention, and it
is single-horizon, not multi-day.

Context retention itself exists only as a TRAINING REWARD, not an evaluation
endpoint: core/rewards/retention.py's reward_context_retention scores whether
must_mention_facts survive into <patient_state>, via substring-or-embedding
matching. It is a coverage fraction in [0,1], not an MAE, and nothing in
evaluation/ currently calls it. A real "context retention MAE over multi-day
windows" endpoint would be new work: it would need multi-day windowed
evaluation data (which core/cohort/mimic.py's 24h-lookback vignettes do not
currently produce) and a numeric retained-vs-true quantity to take an MAE
over. This file is its natural home when that is built.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error


def evaluate_trajectory_accuracy(predicted_values: np.ndarray, true_values: np.ndarray) -> float:
    """Endpoint 3: mean absolute error between forecasted and actually
    observed future values (e.g. predicted vs. true MAP at +6h). Silently
    drops any (predicted, true) pair where EITHER side is NaN -- e.g. a
    variable the model chose not to forecast, or a true value that
    couldn't be computed for that admission -- rather than propagating NaN
    through the whole metric.
    """
    mask = ~np.isnan(predicted_values) & ~np.isnan(true_values)
    if mask.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(true_values[mask], predicted_values[mask]))
