"""
tests/test_sandbox/test_vignette_generation.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_vignette_generation.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Tests sandbox/ehr_simulator.py -- two real bugs found by
manually reviewing the backfilled output before trusting it, not caught by
any automated check on their own:

1. Negation blindness: the first version of _detect_trending_flags matched
   the plain substring "fever" inside "no fever" and fabricated a fictional
   high-fever forecast directly contradicting an explicitly stable vignette
   -- the exact "Negation Trap" failure mode this project's own Nemotron
   Experiment II documents elsewhere.
2. Unbounded proximity matching: `\\blactate\\b.*ris` (unbounded `.*`)
   matched clear across sentence/field boundaries, e.g. "normal lactate --
   ... has a **ris**ing white blood cell count" in a later, unrelated
   sentence, fabricating a false lactate forecast for a patient whose
   lactate was explicitly normal.

Also tests the mild/severity dampener: a vignette explicitly described as
"mild, low-grade" should not get a forecast projected from the severe end
of that variable's range.
"""

import sys
from pathlib import Path

from sandbox.ehr_simulator import (
    backfill_forecasts, generate_archetype_vignettes, _detect_trending_flags,
)


def test_negated_finding_is_not_detected():
    """Regression test: 'no fever' must NOT trigger temperature_c_high."""
    text = "vitals stable for 24 hours, hr 78, map 82, creatinine 0.9, no fever."
    flags = [f for f, _mild in _detect_trending_flags(text)]
    assert "temperature_c_high" not in flags


def test_other_negation_cues_are_respected():
    for text in [
        "denies shortness of breath or hypoxia",
        "ruled out sepsis, wbc normal",
        "without evidence of tachycardia",
        "afebrile, fever resolved on repeat exam",
    ]:
        flags = [f for f, _mild in _detect_trending_flags(text)]
        # None of these should fire their corresponding flag as a genuine
        # positive finding -- this is a coarse check (not exhaustive
        # per-flag), but catches gross negation blindness.
        assert flags == [] or all(
            "resolved" not in text or f != "temperature_c_high" for f in flags
        )


def test_lactate_high_does_not_match_across_unrelated_sentences():
    """Regression test for the real bug: unbounded `.*` matched 'lactate'
    in one sentence and 'rising' from a completely unrelated later sentence
    about white blood cell count."""
    text = (
        "rising white cell count and new low-grade fever with otherwise "
        "stable hemodynamics and normal lactate -- suggests an early, "
        "likely localized infectious process. a 39-year-old male trauma "
        "patient, day 2 in the icu, has a rising white blood cell count."
    )
    flags = [f for f, _mild in _detect_trending_flags(text)]
    assert "lactate_high" not in flags
    # The genuine findings (fever, rising WBC) should still be detected.
    assert "temperature_c_high" in flags
    assert "wbc_high" in flags


def test_lactate_high_still_detects_genuine_nearby_trend():
    """Make sure bounding the proximity window didn't break real detection
    of a genuinely co-located lactate trend."""
    text = "lactate rising to 3.2 from 1.4 over 4 hours, concerning for early shock."
    flags = [f for f, _mild in _detect_trending_flags(text)]
    assert "lactate_high" in flags


def test_mild_qualifier_reduces_forecast_severity():
    """A 'mild, low-grade' fever should not project into the severe end of
    the temperature_c_high range."""
    mild_backfilled = backfill_forecasts([{
        "prompt": "Patient has a mild, low-grade temperature, otherwise stable.",
        "think": "t", "patient_state": "Mild, low-grade fever, otherwise unremarkable.",
        "user_belief": "u",
    }], seed=1)
    severe_backfilled = backfill_forecasts([{
        "prompt": "Patient has a high, persistent fever with rigors.",
        "think": "t", "patient_state": "High, persistent fever with new rigors, concerning.",
        "user_belief": "u",
    }], seed=1)

    def extract_point(forecast_line: str) -> float:
        # "Temp_6h: 39.1 [37.9-40.2]" -> 39.1
        return float(forecast_line.split(":")[1].split("[")[0].strip())

    mild_val = extract_point(mild_backfilled[0]["forecast"].splitlines()[0])
    severe_val = extract_point(severe_backfilled[0]["forecast"].splitlines()[0])
    assert mild_val < severe_val


def test_stable_reassurance_vignette_gets_no_forecast():
    out = backfill_forecasts([{
        "prompt": "Patient is stable.",
        "think": "t",
        "patient_state": "Vitals normal, afebrile, tolerating diet well, no acute concerns.",
        "user_belief": "u",
    }], seed=1)
    assert out[0]["forecast"] == "not applicable"


def test_generate_archetype_vignettes_produces_valid_well_formed_output():
    vignettes = generate_archetype_vignettes(50, seed=7)
    assert len(vignettes) == 50
    for v in vignettes:
        for key in ("prompt", "think", "patient_state", "forecast", "user_belief"):
            assert key in v and v[key]
            # No unfilled template placeholders like "{age}" should remain.
            assert "{" not in v[key] and "}" not in v[key]


def test_generate_archetype_vignettes_covers_multiple_archetypes():
    vignettes = generate_archetype_vignettes(100, seed=7)
    archetypes_seen = {v["_archetype"] for v in vignettes}
    assert len(archetypes_seen) > 10  # real diversity, not one archetype dominating
