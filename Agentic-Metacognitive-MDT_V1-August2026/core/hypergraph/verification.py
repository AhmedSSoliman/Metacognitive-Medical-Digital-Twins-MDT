"""
core/hypergraph/verification.py

Ported unchanged from the source repo's hypergraph/verification.py
(../Agentic-DT_V1-July/hypergraph/verification.py). No intra-project imports,
so nothing needed rewriting. References below to "training/rewards.py" now
correspond to core/rewards/boundary.py (R_bound) in this layout, and
"hypergraph/construction.py" to core/hypergraph/construction.py.

Two implementations of the R_bound safety-check interface used by
training/rewards.py:

  - InterimRuleBasedChecker: a small, hand-specified set of physiologically
    implausible claim patterns (e.g. "bradycardic and tachycardic at the
    same time"), used from the start of Phase 2 before the Phase 3
    hypergraph exists.
  - LearnedHypergraphChecker: checks a generated <patient_state> against the
    mined-and-clinically-reviewed hyperedges from hypergraph/construction.py.

Both expose the same `.check(patient_state_text) -> float in [0, 1]`
interface so Phase 2 training code does not need to change when swapping
one for the other.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interim rule-based checker (available from day one, no MIMIC data needed)
# ---------------------------------------------------------------------------

# Each rule: (pattern_a, pattern_b, is_implausible_if_both_present)
_IMPLAUSIBLE_PAIRS = [
    (r"\bbradycardi", r"\btachycardi", True),          # can't be both slow and fast HR
    (r"\bhypertensiv", r"\bhypotensiv", True),          # can't be both high and low BP simultaneously
    (r"\bhyperthermi|\bfebrile", r"\bhypothermi", True),
    (r"\bhyperkalemi", r"\bhypokalemi", True),
    (r"\bpolyuri", r"\banuri|\boliguri", True),
]

# Physiologically PLAUSIBLE co-occurrences (should not be flagged, sanity list
# for testing the checker itself, not used in scoring)
_PLAUSIBLE_EXAMPLES = [
    "tachycardic and hypotensive",       # classic shock physiology
    "tachypneic and hypoxic",
]


class InterimRuleBasedChecker:
    """Zero-training-data checker: flags a small set of hand-specified
    physiologically impossible or highly implausible co-occurring claims.
    This is intentionally conservative (few rules, high precision) since it
    is meant to be a safety floor, not a comprehensive model of physiology --
    the Phase 3 LearnedHypergraphChecker is meant to supersede it.
    """

    def __init__(self):
        self.rules = _IMPLAUSIBLE_PAIRS

    def check(self, patient_state_text: str) -> float:
        if not patient_state_text:
            return 0.0
        text = patient_state_text.lower()
        violations = 0
        for pattern_a, pattern_b, _ in self.rules:
            if re.search(pattern_a, text) and re.search(pattern_b, text):
                violations += 1
        if violations == 0:
            return 1.0
        # Each violation is a hard penalty -- these are meant to be rare and clear-cut
        return max(0.0, 1.0 - 0.5 * violations)


# ---------------------------------------------------------------------------
# Learned hypergraph checker (Phase 3, requires clinically-reviewed hyperedges)
# ---------------------------------------------------------------------------

# Maps hyperedge variable-flag names (e.g. "heart_rate_high") to text patterns
# that would indicate the SAME abnormality is being claimed in free text.
# Extend this as the hypergraph's variable vocabulary grows.
_FLAG_TO_TEXT_PATTERNS = {
    "heart_rate_high": [r"\btachycardi", r"\belevated heart rate", r"\bHR (?:of )?1\d\d"],
    "heart_rate_low": [r"\bbradycardi"],
    "map_low": [r"\bhypotensi", r"\blow (?:blood pressure|MAP)", r"\bMAP (?:of )?[0-5]\d\b"],
    "sbp_low": [r"\bhypotensi"],
    "resp_rate_high": [r"\btachypne"],
    "resp_rate_low": [r"\bbradypne"],
    "spo2_low": [r"\bhypoxi", r"\bdesaturat"],
    "temperature_c_high": [r"\bfebrile", r"\bfever", r"\bhyperthermi"],
    "temperature_c_low": [r"\bhypothermi"],
    "lactate_high": [r"\bhyperlactatemi", r"\belevated lactate", r"\blactate (?:of |is )?[2-9]"],
    "creatinine_high": [r"\brenal (?:dysfunction|failure|injury)", r"\belevated creatinine", r"\bAKI\b"],
    "wbc_high": [r"\bleukocytosis"],
    "wbc_low": [r"\bleukopeni"],
    "platelet_low": [r"\bthrombocytopeni"],
    "bilirubin_total_high": [r"\bhyperbilirubinemi", r"\bjaundic"],
    "ph_arterial_low": [r"\bacidosis|\bacidemi"],
}


class LearnedHypergraphChecker:
    """Loads a clinically-reviewed hypergraph (JSON produced by
    hypergraph.construction.build_and_save_hypergraph, after review) and
    checks whether a generated <patient_state> claims a combination of
    abnormalities that is EITHER (a) consistent with at least one known
    hyperedge, or (b) does not claim enough simultaneous abnormalities to
    be checkable at all (neutral score).

    A LOW score is returned when the text claims a combination of >= 2
    abnormalities that does NOT match any reviewed hyperedge -- this is the
    trainable safety signal for R_bound.
    """

    def __init__(self, hypergraph_path: str, require_reviewed: bool = True):
        with open(hypergraph_path) as f:
            data = json.load(f)

        if require_reviewed and data.get("status") != "CLINICALLY_REVIEWED":
            raise ValueError(
                f"Hypergraph at {hypergraph_path} has status '{data.get('status')}', "
                f"not 'CLINICALLY_REVIEWED'. Refusing to use unreviewed hyperedges as a "
                f"safety constraint -- either complete the clinical face-validity review "
                f"and update the status field, or pass require_reviewed=False explicitly "
                f"if you understand the risk (e.g. for early-stage debugging only)."
            )

        self.hyperedges = [frozenset(h["variables"]) for h in data["hyperedges"]]
        logger.info("Loaded %d reviewed hyperedges from %s", len(self.hyperedges), hypergraph_path)

    def _detect_flags(self, text: str) -> set[str]:
        text = text.lower()
        detected = set()
        for flag, patterns in _FLAG_TO_TEXT_PATTERNS.items():
            if any(re.search(p, text) for p in patterns):
                detected.add(flag)
        return detected

    def check(self, patient_state_text: str) -> float:
        if not patient_state_text:
            return 0.0
        detected_flags = self._detect_flags(patient_state_text)
        if len(detected_flags) < 2:
            return 1.0  # not enough simultaneous claims to be checkable -- neutral, not penalized

        # Does the detected flag set match (is a superset or subset of) ANY known hyperedge?
        for hyperedge in self.hyperedges:
            if hyperedge.issubset(detected_flags) or detected_flags.issubset(hyperedge):
                return 1.0

        # No matching hyperedge for this combination -- flag as physiologically
        # ungrounded (not necessarily FALSE, but unsupported by the derived hypergraph)
        return 0.2


if __name__ == "__main__":
    checker = InterimRuleBasedChecker()
    tests = [
        ("Patient is tachycardic and hypotensive, consistent with early shock.", 1.0),
        ("Patient is bradycardic and tachycardic simultaneously.", 0.5),
        ("Patient is normotensive with stable vitals.", 1.0),
    ]
    for text, expected in tests:
        score = checker.check(text)
        status = "OK" if abs(score - expected) < 1e-6 else "MISMATCH"
        print(f"[{status}] score={score:.2f} (expected {expected}): {text}")
