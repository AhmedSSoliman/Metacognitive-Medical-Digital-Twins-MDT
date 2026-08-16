"""
sandbox/ehr_simulator.py

Synthetic patient-vignette generation across ~25 clinical archetypes, plus
forecast backfilling onto hand-written vignettes.

PORTING NOTE (2026-08-12): ported BYTE-IDENTICAL (apart from this header)
from the source repo's data/synthetic/vignette_generation.py
(../Agentic-DT_V1-July/data/synthetic/vignette_generation.py). It has no
intra-project imports, so nothing needed rewriting.

=== JUDGMENT CALL: WHY THIS FILE IS HERE, AND WHAT IT IS NOT ===

The task spec asked whether this module is a genuine synthetic-patient-
trajectory generator fit for integration-test sandboxing, or cohort
generation for training. Having read it in full: it is the LATTER, and this
placement is a compromise that needs stating plainly so nobody mistakes it
for something it isn't.

WHAT IT ACTUALLY IS. This generates Phase 1 SFT TRAINING DATA. Its own
docstring (preserved below) is explicit: it exists to fix two defects in the
training mix -- 36 hand-written vignettes all missing a real `forecast`
field, and only 36 unique examples requiring ~48x oversampling. Its output is
a JSONL of {prompt, think, patient_state, forecast, user_belief} records fed
to training/sft.py's format_synthetic_example. Those five fields are a
TRAINING TARGET: each record already contains the model's ideal ANSWER, not
just a patient situation.

WHY THAT IS NOT AN EHR SANDBOX. A virtual EHR sandbox would emit patient
STATE that a system under test then reasons about -- time-series vitals/labs
over an admission, queryable at a cutoff, with ground truth held back for
scoring. This module emits no time series at all: each vignette is a single
static text snapshot with pre-written reasoning attached. There are no
trajectories to step through, nothing to query, and no held-back ground
truth (the answer is in the record).

WHAT WOULD ACTUALLY BE PORTED IF A SANDBOX EXISTED. Nothing -- confirmed by
grepping the whole source repo for 'sandbox', 'simulator', 'end_to_end', and
'end-to-end': the only matches are the words "end-to-end" and "simulate"
inside unrelated comments in tests/test_stopping_criteria.py,
scripts/run_evaluation.py, and evaluation/metrics.py. No EHR sandbox exists.

WHY IT WAS STILL PLACED HERE rather than left in a data/ module: it is the
only synthetic-patient generator in the project, and it already contains the
two pieces a real sandbox would reuse -- the archetype/value sampling
machinery (_sample_archetype_vars, ARCHETYPES) and the abnormality-flag
detection vocabulary shared with core/hypergraph/verification.py. Building
sandbox/ehr_simulator.py for real means extending THIS code to emit
timestamped series instead of static text, so it is the honest starting
point. Note the consequence for the dependency layering: training/sft.py's
DATA depends on this file, so a training data-prep path now reaches into
sandbox/. If that coupling becomes a problem, the right fix is to move the
archetype tables back under core/cohort/ and leave a thin trajectory-emitting
wrapper here.

Its CLI driver (the source repo's data/build_synthetic_vignettes.py) is
ported alongside as sandbox/build_synthetic_vignettes.py, and its tests as
tests/test_sandbox/test_vignette_generation.py.

ORIGINAL data/synthetic/vignette_generation.py MODULE DOCSTRING:

    Programmatic generator for Phase 1 SFT stream-format vignettes
    (data/synthetic/stream_format_vignettes.jsonl), covering two real gaps
    found while testing a trained checkpoint:

    1. All 36 original hand-written vignettes were missing the `forecast`
       field entirely (confirmed by direct inspection -- every one silently
       defaulted to "not applicable" via training/sft.py's
       `ex.get("forecast", "not applicable")`). Combined with the tier-one
       reasoning dataset ALSO always using "not applicable" for this field,
       Phase 1's SFT training data taught the model "not applicable" is always
       correct for <forecast> -- 100% of examples, 0% ever demonstrating the
       real numeric `VAR_Nh: value [low-high]` sub-format -- which fully
       explains a real generation where the model never attempted a forecast
       even for an ICU deterioration-risk vignette that clearly called for one.
    2. Only 36 unique vignettes existed, requiring ~48x oversampling to reach
       an 8% target training-mix fraction (training/sft.py's
       `load_sft_dataset`) -- right at the edge of that function's own
       50x memorization-risk warning threshold.

    This module provides:
      - `backfill_forecasts(vignettes)`: adds a real forecast field to
        existing hand-written vignettes by inferring the actively-trending
        physiological variable(s) from their own patient_state text (reusing
        core/hypergraph/verification.py's flag-detection patterns, not fragile new
        regex), and projecting a clinically plausible 6h-ahead value.
      - `generate_archetype_vignettes(n)`: generates N new, fully-synthetic
        vignettes across ~25 clinical archetypes, each independently sampled
        (demographics, exact values, recipient type, phrasing variant) so a
        large batch is genuinely diverse, not near-duplicate text -- see
        ARCHETYPES below.

    Forecast variable naming matches core/hypergraph/construction.py's
    ABNORMALITY_RULES vocabulary (heart_rate, map, lactate, creatinine, etc.)
    for the core hemodynamic/deterioration archetypes, so this stays
    consistent with the rest of the pipeline's variable vocabulary; other
    archetypes use domain-specific variable names (e.g. potassium, inr) where
    a real quantitative forecast is clinically meaningful, or correctly
    "not applicable" where it genuinely is not (e.g. a stable reassurance case).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Part 1: backfilling forecasts onto the existing 36 hand-written vignettes
# ---------------------------------------------------------------------------

# Reuses the exact same flag/text-pattern vocabulary as
# core/hypergraph/verification.py's LearnedHypergraphChecker, so "what variable
# is this vignette actually about" is answered the same way training-time
# reward scoring would answer it -- not a separate, potentially-inconsistent
# heuristic.
_FLAG_TO_FORECAST = {
    # flag_name: (forecast_var_name, direction, plausible_6h_value_range, interval_half_width_fraction)
    "heart_rate_high": ("HR_6h", "worsening", (115, 145), 0.12),
    "heart_rate_low": ("HR_6h", "worsening", (35, 55), 0.15),
    "map_low": ("MAP_6h", "worsening", (48, 60), 0.12),
    "sbp_low": ("SBP_6h", "worsening", (75, 88), 0.12),
    "resp_rate_high": ("RR_6h", "worsening", (26, 34), 0.15),
    "resp_rate_low": ("RR_6h", "worsening", (6, 9), 0.2),
    "spo2_low": ("SpO2_6h", "worsening", (84, 90), 0.06),
    "temperature_c_high": ("Temp_6h", "worsening", (38.8, 40.0), 0.03),
    "temperature_c_low": ("Temp_6h", "worsening", (34.5, 35.8), 0.04),
    "lactate_high": ("lactate_6h", "worsening", (3.2, 5.5), 0.2),
    "creatinine_high": ("creatinine_6h", "worsening", (1.8, 3.0), 0.2),
    "wbc_high": ("WBC_6h", "worsening", (14.0, 20.0), 0.2),
    "wbc_low": ("WBC_6h", "worsening", (1.0, 3.0), 0.25),
    "platelet_low": ("platelet_6h", "worsening", (40, 85), 0.2),
    "bilirubin_total_high": ("bilirubin_6h", "worsening", (3.0, 6.0), 0.2),
    "ph_arterial_low": ("pH_6h", "worsening", (7.15, 7.30), 0.02),
}

_FLAG_TO_TEXT_PATTERNS = {
    "heart_rate_high": [r"\btachycardi", r"\bHR (?:of )?1\d\d"],
    "heart_rate_low": [r"\bbradycardi"],
    "map_low": [r"\bhypotensi", r"\bMAP (?:of )?[0-5]\d\b"],
    "sbp_low": [r"\bhypotensi", r"\bSBP (?:of )?[0-8]\d\b"],
    "resp_rate_high": [r"\btachypne"],
    "resp_rate_low": [r"\bbradypne"],
    "spo2_low": [r"\bhypoxi", r"\bdesaturat", r"\boxygen requirement increase"],
    "temperature_c_high": [r"\bfebrile", r"\bfever", r"\bhyperthermi", r"\btemperature of 3[89]\."],
    "temperature_c_low": [r"\bhypothermi"],
    # NOTE: proximity windows (.{0,N}) are deliberately BOUNDED, not the
    # unbounded `.*` these three originally used -- a real bug caught by
    # manual review: `.*` matched clear across sentence/field boundaries
    # (e.g. "normal lactate -- ... has a **ris**ing white blood cell count"
    # in an unrelated later sentence incorrectly matched "lactate.*ris" for
    # a patient whose lactate was explicitly normal, fabricating a false
    # lactate forecast). Bounding to a short window keeps the two words
    # close enough to plausibly describe the same clinical fact.
    "lactate_high": [r"\bhyperlactatemi", r"\blactate\b.{0,20}ris", r"\blactate that rose"],
    "creatinine_high": [r"\brenal (?:dysfunction|failure|injury)", r"\bcreatinine\b.{0,20}ros", r"\bAKI\b"],
    "wbc_high": [r"\bleukocytosis", r"\bwhite blood cell count\b.{0,20}ris", r"\brising white blood"],
    "wbc_low": [r"\bleukopeni", r"\bneutropeni", r"\bANC \d"],
    "platelet_low": [r"\bthrombocytopeni"],
    "bilirubin_total_high": [r"\bhyperbilirubinemi", r"\bjaundic"],
    "ph_arterial_low": [r"\bacidosis\b", r"\bacidemi"],
}



# A real bug caught by manual review, not hypothetical: the first version of
# this function had no negation handling, so a vignette explicitly stating
# "no fever" matched the plain substring "fever" and fabricated a fictional
# high-fever forecast directly contradicting the vignette's own text -- the
# exact "Negation Trap" failure mode this project's own Nemotron Experiment
# II documents and studies elsewhere (naive keyword matching flagging
# negated findings as positive). This is a lightweight local guard, not a
# full NegEx/dependency-parse implementation -- sufficient for vignette
# generation-time inference (much lower stakes than clinical evaluation
# scoring), but not a substitute for real negation detection if reused
# elsewhere.
_PRE_NEGATION_CUES = [
    "no ", "not ", "without ", "denies ", "denied ", "negative for ",
    "ruled out ", "rule out ", "absence of ",
]
# Some negation phrasing puts the resolution cue AFTER the term instead of
# before it (e.g. "fever resolved", "confusion has resolved") -- a
# backward-only window misses these. Checked as a separate forward window
# rather than folded into _PRE_NEGATION_CUES, since these specific cues
# only count as negation when they follow the term, not precede it (e.g.
# "resolving the underlying infection caused the fever" is NOT a negation
# despite containing "resolving").
_POST_NEGATION_CUES = ["resolved", "resolving", "has cleared", "normalized", "back to baseline"]


def _is_negated(text: str, match_start: int, match_end: int = None, window: int = 25) -> bool:
    match_end = match_end if match_end is not None else match_start
    preceding = text[max(0, match_start - window):match_start]
    following = text[match_end:match_end + window]
    return (any(cue in preceding for cue in _PRE_NEGATION_CUES)
            or any(cue in following for cue in _POST_NEGATION_CUES))


_MILD_QUALIFIER_CUES = ["mild", "low-grade", "borderline", "slight", "minimal", "benign", "trivial"]


def _is_mild_context(text: str, match_start: int, match_end: int, window: int = 40) -> bool:
    """Checks for a mild/low-grade qualifier near the match. Catches a real
    calibration bug found by manual review: without this, a vignette
    explicitly described as "mild, low-grade... common and typically
    benign" got a projected forecast in the SEVERE end of the variable's
    range (e.g. a 39.9C high-fever prediction for a documented low-grade
    fever) -- factually consistent with "some fever present" but wildly
    inconsistent with the vignette's own stated severity."""
    surrounding = text[max(0, match_start - window):match_end + window]
    return any(cue in surrounding for cue in _MILD_QUALIFIER_CUES)


def _detect_trending_flags(text: str) -> list[tuple[str, bool]]:
    """Returns [(flag, is_mild_context), ...]."""
    import re
    text = text.lower()
    detected = []
    for flag, patterns in _FLAG_TO_TEXT_PATTERNS.items():
        for p in patterns:
            m = re.search(p, text)
            if m and not _is_negated(text, m.start(), m.end()):
                detected.append((flag, _is_mild_context(text, m.start(), m.end())))
                break
    return detected


def _project_forecast_line(flag: str, rng: random.Random, mild: bool = False) -> str:
    var, _, (lo_range, hi_range), half_width_frac = _FLAG_TO_FORECAST[flag]
    if mild:
        # Sample only the bottom quartile of the abnormal range instead of
        # its full span, so a documented mild/low-grade finding doesn't get
        # projected as if it were the severe end of that abnormality.
        hi_range = lo_range + 0.25 * (hi_range - lo_range)
    point = rng.uniform(lo_range, hi_range)
    half_width = max(point * half_width_frac, 0.05)
    low, high = point - half_width, point + half_width
    decimals = 2 if "pH" in var else (1 if any(k in var for k in ("Temp", "lactate", "creatinine", "bilirubin")) else 0)
    fmt = f"%.{decimals}f"
    return f"{var}: {fmt % point} [{fmt % low}-{fmt % high}]"


def backfill_forecasts(vignettes: list[dict], seed: int = 42) -> list[dict]:
    """Returns a NEW list with a real `forecast` field added to each
    vignette, inferred from its own patient_state (falling back to prompt
    text if patient_state alone doesn't surface a clear trending flag).
    Vignettes with no detectable trending abnormality (a genuinely stable/
    reassurance case) correctly keep "not applicable" -- this does not
    fabricate a forecast where none is clinically warranted.
    """
    rng = random.Random(seed)
    out = []
    for v in vignettes:
        v = dict(v)
        combined_text = f"{v.get('patient_state', '')} {v.get('prompt', '')}"
        flags = _detect_trending_flags(combined_text)
        flags = [(f, mild) for f, mild in flags if f in _FLAG_TO_FORECAST]
        if not flags:
            v["forecast"] = "not applicable"
        else:
            # Cap at 2 forecast lines -- a vignette flagging 4+ abnormalities
            # doesn't need every single one forecast to demonstrate the format.
            chosen = flags[:2]
            v["forecast"] = "\n".join(_project_forecast_line(f, rng, mild=mild) for f, mild in chosen)
        out.append(v)
    return out


# ---------------------------------------------------------------------------
# Part 2: generating new, fully-synthetic vignettes from archetypes
# ---------------------------------------------------------------------------

@dataclass
class Archetype:
    name: str
    contexts: list[str]              # admission/presentation context templates, {age}/{gender} filled in
    think_templates: list[str]       # reasoning text templates, {vars} filled in from sampled values
    patient_state_templates: list[str]
    forecast_vars: list[str]         # which _FLAG_TO_FORECAST-style vars this archetype projects, or [] for none
    forecast_ranges: dict            # var -> (lo, hi, half_width_frac, decimals)
    urgency: str                     # "urgent" | "moderate" | "reassuring" -- informs phrasing tone


def _sample_demographics(rng: random.Random) -> dict:
    age = rng.randint(19, 88)
    gender = rng.choice(["male", "female"])
    return {"age": age, "gender": gender}


_RECIPIENT_FRAMINGS = {
    "clinician": [
        "Likely read by the covering {role}; keep the reasoning clinically concrete and flag the urgency level clearly.",
        "This will be reviewed by the primary team on rounds; use precise clinical terminology and note the trend explicitly.",
        "Addressed to the {role} managing this patient; be direct about the differential and next steps.",
    ],
    "patient": [
        "The patient themselves is asking; avoid jargon, explain what the numbers mean in plain terms, and be honest without being alarming.",
        "This is for the patient directly; use accessible language and acknowledge their likely anxiety about the result.",
        "Patient-facing explanation; define any clinical terms used and keep the tone calm and clear.",
    ],
    "family": [
        "The patient's family is asking, and they have limited medical background; frame this in terms of what to expect next, not raw numbers.",
        "Addressed to anxious family members at the bedside; be reassuring where genuinely warranted but honest about uncertainty.",
        "Family meeting context; avoid clinical jargon and focus on what this means for the patient's course.",
    ],
}

_ROLES = ["night float resident", "ICU nurse", "primary team", "surgical resident", "hospitalist", "rapid response team"]


ARCHETYPES: list[Archetype] = [
    Archetype(
        name="post_op_septic_hypovolemic_shock",
        contexts=[
            "A {age}-year-old {gender}, {hours}h post-op from a {surgery}, has a heart rate of {hr}, MAP of {map}, "
            "and a lactate that rose from {lac_start} to {lac_now} over the last {trend_hours}h.",
        ],
        think_templates=[
            "HR climbing to {hr}, MAP dropping to {map}, and lactate rising from {lac_start} to {lac_now} together over "
            "a short window -- this triad is consistent with early septic or hypovolemic shock rather than isolated "
            "pain-related tachycardia. Worth also considering post-op bleeding, which can produce the same triad "
            "without fever. Given the trajectory, escalation of monitoring is warranted now, not after further decline.",
        ],
        patient_state_templates=[
            "Tachycardic (HR {hr}), borderline hypotensive (MAP {map}), rising lactate ({lac_now}, up from {lac_start}) "
            "-- consistent with early shock of undetermined source (septic vs. hemorrhagic); recommend urgent "
            "reassessment and repeat labs within 1-2 hours.",
        ],
        forecast_vars=["MAP_6h", "lactate_6h"],
        forecast_ranges={"MAP_6h": (48, 60, 0.1, 0), "lactate_6h": (3.2, 5.0, 0.2, 1)},
        urgency="urgent",
    ),
    Archetype(
        name="dka",
        contexts=[
            "A {age}-year-old {gender} with type {dm_type} diabetes was admitted for DKA. Current glucose {glucose}, "
            "pH {ph}, and {improving_or_not}.",
        ],
        think_templates=[
            "Glucose at {glucose} and pH {ph} -- {ph_interpretation}. The anion gap trend and clinical exam matter as "
            "much as the raw pH here; if closing, this is a genuine improvement, not just a number moving in the "
            "right direction on paper.",
        ],
        patient_state_templates=[
            "DKA under treatment: glucose {glucose}, pH {ph} ({ph_interpretation}); continue current insulin/fluid "
            "protocol and recheck labs per DKA pathway.",
        ],
        forecast_vars=["pH_6h"],
        forecast_ranges={"pH_6h": (7.28, 7.40, 0.015, 2)},
        urgency="moderate",
    ),
    Archetype(
        name="respiratory_decompensation",
        contexts=[
            "A {age}-year-old admitted with {resp_dx} now has an oxygen requirement increase from {o2_start} to "
            "{o2_now} over the last {trend_hours}h, with SpO2 {spo2} on current support.",
        ],
        think_templates=[
            "Escalating oxygen requirement ({o2_start} -> {o2_now}) with SpO2 {spo2} despite support is a real "
            "deterioration signal, not noise -- this pattern in {resp_dx} raises concern for progression toward "
            "needing higher-level respiratory support if the trend continues.",
        ],
        patient_state_templates=[
            "Worsening hypoxia (SpO2 {spo2}) with escalating oxygen requirement ({o2_start} to {o2_now}) in the "
            "setting of {resp_dx}; recommend repeat ABG and close respiratory monitoring, with a low threshold for "
            "escalating support.",
        ],
        forecast_vars=["SpO2_6h"],
        forecast_ranges={"SpO2_6h": (84, 91, 0.05, 0)},
        urgency="urgent",
    ),
    Archetype(
        name="stable_post_op_reassurance",
        contexts=[
            "A {age}-year-old {gender} is {days} days post {surgery}. Vitals and labs have been stable for 24 hours: "
            "HR {hr}, MAP {map}, temperature {temp}C, and pain is well-controlled.",
        ],
        think_templates=[
            "Vitals (HR {hr}, MAP {map}, temp {temp}C) are all within normal range and have been stable for a full "
            "24h -- this is a reassuring, unremarkable post-op course with nothing pointing toward an evolving "
            "complication at this time.",
        ],
        patient_state_templates=[
            "Hemodynamically stable (HR {hr}, MAP {map}), afebrile (temp {temp}C), with well-controlled pain and no "
            "signs of surgical complication; continue routine post-op monitoring.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="reassuring",
    ),
    Archetype(
        name="rising_infection_marker",
        contexts=[
            "A {age}-year-old {gender}, day {days} in the {unit}, has a rising white blood cell count ({wbc_start} to "
            "{wbc_now}) and a new low-grade fever of {temp}C.",
        ],
        think_templates=[
            "WBC climbing from {wbc_start} to {wbc_now} with a new fever ({temp}C) suggests an evolving infectious "
            "process rather than a post-surgical inflammatory response alone -- worth a focused source workup "
            "(wound, line, urine, lungs) rather than waiting for further trend confirmation.",
        ],
        patient_state_templates=[
            "New leukocytosis (WBC {wbc_now}, up from {wbc_start}) with low-grade fever ({temp}C) -- possible "
            "evolving infection; recommend source workup (cultures, wound check, imaging as indicated).",
        ],
        forecast_vars=["WBC_6h"],
        forecast_ranges={"WBC_6h": (14.0, 19.0, 0.15, 1)},
        urgency="moderate",
    ),
    Archetype(
        name="hyperkalemia",
        contexts=[
            "A {age}-year-old {gender} post-{surgery} has a potassium of {k_now} (up from {k_start} yesterday) with "
            "{ecg_finding} on ECG.",
        ],
        think_templates=[
            "Potassium rising to {k_now} with {ecg_finding} is a genuine cardiac-risk finding, not just a lab "
            "abnormality to trend -- this needs treatment now (stabilization plus a shift/elimination strategy), "
            "not repeat-and-wait.",
        ],
        patient_state_templates=[
            "Hyperkalemia (K {k_now}, up from {k_start}) with {ecg_finding} -- treat as a cardiac emergency: "
            "calcium for membrane stabilization, insulin/glucose or other shifting agent, and address the "
            "underlying cause.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="urgent",
    ),
    Archetype(
        name="postpartum_stable_reassurance",
        contexts=[
            "A {age}-year-old woman, day {days} after an uncomplicated {delivery_type} delivery, has normal vitals "
            "(HR {hr}, MAP {map}) and is breastfeeding without difficulty.",
        ],
        think_templates=[
            "Vitals are normal (HR {hr}, MAP {map}) and the postpartum course sounds unremarkable so far -- nothing "
            "here suggests hemorrhage, infection, or another evolving complication.",
        ],
        patient_state_templates=[
            "Uncomplicated postpartum course, hemodynamically normal (HR {hr}, MAP {map}), tolerating breastfeeding; "
            "routine postpartum monitoring, no acute concerns identified.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="reassuring",
    ),
    Archetype(
        name="hepatic_decompensation",
        contexts=[
            "A {age}-year-old {gender} with cirrhosis has an INR that increased from {inr_start} to {inr_now} over "
            "{trend_hours}h along with {mental_status}.",
        ],
        think_templates=[
            "INR rising from {inr_start} to {inr_now} together with {mental_status} points toward worsening hepatic "
            "synthetic function and possible early hepatic encephalopathy -- these two findings together are more "
            "concerning than either alone.",
        ],
        patient_state_templates=[
            "Worsening hepatic decompensation: INR {inr_now} (up from {inr_start}), {mental_status} -- recommend "
            "ammonia level, lactulose titration, and reassessment of transplant/escalation candidacy status.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="urgent",
    ),
    Archetype(
        name="mild_exacerbation_improving",
        contexts=[
            "A {age}-year-old {gender} with well-controlled {chronic_dx} is admitted for observation after a mild "
            "exacerbation, now on room air with {symptom} resolved.",
        ],
        think_templates=[
            "Off supplemental oxygen and {symptom} has resolved -- this is a genuinely reassuring trajectory for "
            "a mild {chronic_dx} exacerbation, consistent with a routine, uncomplicated recovery.",
        ],
        patient_state_templates=[
            "Mild {chronic_dx} exacerbation now resolving: room air, {symptom} resolved; appropriate for continued "
            "observation with a plan toward discharge if this trend holds.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="reassuring",
    ),
    Archetype(
        name="acs_chest_pain",
        contexts=[
            "A {age}-year-old {gender} with known {cardiac_hx} reports new central chest pressure radiating to the "
            "left arm, with diaphoresis and nausea, HR {hr}, and a first troponin of {trop_now}.",
        ],
        think_templates=[
            "Classic anginal-equivalent presentation (chest pressure, arm radiation, diaphoresis, nausea) in someone "
            "with {cardiac_hx} plus an elevated first troponin ({trop_now}) is high pretest probability for ACS -- "
            "this needs urgent cardiology involvement and serial troponins, not a wait-and-see approach.",
        ],
        patient_state_templates=[
            "Presentation highly concerning for ACS: classic anginal symptoms, HR {hr}, troponin {trop_now} "
            "(elevated); recommend urgent ECG review, serial troponins, and cardiology consultation.",
        ],
        forecast_vars=["troponin_6h"],
        forecast_ranges={"troponin_6h": (0.8, 3.5, 0.25, 2)},
        urgency="urgent",
    ),
    Archetype(
        name="copd_bipap_improving",
        contexts=[
            "A {age}-year-old {gender} with COPD exacerbation has been on BiPAP for {bipap_hours}h. ABG shows pCO2 "
            "improved from {pco2_start} to {pco2_now}, and work of breathing has decreased.",
        ],
        think_templates=[
            "pCO2 improving from {pco2_start} to {pco2_now} with decreased work of breathing on BiPAP is a genuinely "
            "positive response to non-invasive support -- the trajectory argues against needing intubation if this "
            "trend continues.",
        ],
        patient_state_templates=[
            "COPD exacerbation responding well to BiPAP: pCO2 improving ({pco2_start} -> {pco2_now}), decreased work "
            "of breathing; continue current support with a plan to trial weaning if the trend holds.",
        ],
        forecast_vars=["pCO2_6h"],
        forecast_ranges={"pCO2_6h": (48, 58, 0.08, 0)},
        urgency="reassuring",
    ),
    Archetype(
        name="renal_function_decline",
        contexts=[
            "A {age}-year-old {gender} with CKD stage {ckd_stage}, admitted for {infection_dx}, has a creatinine "
            "that rose from {cr_start} to {cr_now} over {trend_hours}h.",
        ],
        think_templates=[
            "Creatinine rising from {cr_start} to {cr_now} in the setting of {infection_dx} and pre-existing CKD "
            "stage {ckd_stage} raises concern for acute-on-chronic kidney injury -- likely from a combination of "
            "the acute illness and any nephrotoxic exposures (contrast, certain antibiotics); review the medication "
            "list and volume status now rather than after further decline.",
        ],
        patient_state_templates=[
            "Acute-on-chronic kidney injury: creatinine {cr_now} (up from {cr_start}) in CKD stage {ckd_stage} with "
            "{infection_dx}; recommend nephrotoxin review, volume status assessment, and close creatinine trending.",
        ],
        forecast_vars=["creatinine_6h"],
        forecast_ranges={"creatinine_6h": (1.8, 2.8, 0.15, 1)},
        urgency="moderate",
    ),
    Archetype(
        name="stable_new_arrhythmia",
        contexts=[
            "A {age}-year-old {gender} with newly diagnosed atrial fibrillation, rate now controlled (HR {hr}, "
            "irregular), asks about what this means going forward.",
        ],
        think_templates=[
            "Rate is now controlled at HR {hr} -- the acute issue is addressed; the remaining conversation is about "
            "anticoagulation decisions and long-term rhythm management, not an acute deterioration concern.",
        ],
        patient_state_templates=[
            "New atrial fibrillation, rate-controlled (HR {hr}, irregular); hemodynamically stable. Ongoing "
            "management focus: anticoagulation risk assessment and rate vs. rhythm control strategy.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="reassuring",
    ),
    Archetype(
        name="post_op_delirium",
        contexts=[
            "A {age}-year-old {gender} post-{surgery}, day {days}, has new mild confusion, is intermittently "
            "inattentive, and {sleep_wake}.",
        ],
        think_templates=[
            "New confusion and inattention post-op, with {sleep_wake}, is consistent with post-operative delirium -- "
            "the priority is a focused workup for reversible contributors (infection, medications, metabolic "
            "derangement, pain control) rather than assuming this is simply expected post-op confusion.",
        ],
        patient_state_templates=[
            "Likely post-operative delirium: new confusion, inattention, {sleep_wake}; recommend delirium workup "
            "(infection screen, medication review, metabolic panel) and non-pharmacologic reorientation measures.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="moderate",
    ),
    Archetype(
        name="pe_suspected",
        contexts=[
            "A {age}-year-old {gender} with a history of {pe_risk}, now {days} days post {surgery}, has new sudden "
            "onset shortness of breath, HR {hr}, and SpO2 {spo2}.",
        ],
        think_templates=[
            "Sudden-onset dyspnea with tachycardia (HR {hr}) and hypoxia (SpO2 {spo2}) in a patient with {pe_risk} "
            "and recent {surgery} is a classic PE presentation -- this needs urgent imaging (CTPA) and empiric "
            "anticoagulation consideration while workup is pending, not a delayed workup.",
        ],
        patient_state_templates=[
            "High suspicion for pulmonary embolism: sudden dyspnea, HR {hr}, SpO2 {spo2}, in a patient with "
            "{pe_risk} and recent surgery; recommend urgent CTPA and consideration of empiric anticoagulation "
            "pending confirmation.",
        ],
        forecast_vars=["SpO2_6h", "HR_6h"],
        forecast_ranges={"SpO2_6h": (85, 91, 0.05, 0), "HR_6h": (115, 135, 0.1, 0)},
        urgency="urgent",
    ),
    Archetype(
        name="variceal_gi_bleed",
        contexts=[
            "A {age}-year-old {gender} with cirrhosis and known esophageal varices develops sudden large-volume "
            "hematemesis. HR {hr}, SBP {sbp}.",
        ],
        think_templates=[
            "Sudden large-volume hematemesis with tachycardia (HR {hr}) and hypotension (SBP {sbp}) in a patient "
            "with known varices is a hemorrhagic emergency -- this needs immediate resuscitation, urgent GI "
            "consultation for endoscopy, and consideration of vasoactive medication, not a wait-and-monitor approach.",
        ],
        patient_state_templates=[
            "Acute variceal hemorrhage: HR {hr}, SBP {sbp}, active large-volume hematemesis; recommend immediate "
            "resuscitation (IV access, blood products as needed), urgent endoscopy, and vasoactive medication.",
        ],
        forecast_vars=["SBP_6h", "HR_6h"],
        forecast_ranges={"SBP_6h": (75, 88, 0.1, 0), "HR_6h": (118, 140, 0.1, 0)},
        urgency="urgent",
    ),
    Archetype(
        name="gi_bleed_anticoagulated",
        contexts=[
            "A {age}-year-old {gender} with a history of GI bleed on anticoagulation for {anticoag_reason} now has "
            "melena and a hemoglobin that dropped from {hgb_start} to {hgb_now}.",
        ],
        think_templates=[
            "Melena with hemoglobin dropping from {hgb_start} to {hgb_now} in someone anticoagulated for "
            "{anticoag_reason} is a genuine bleeding event, not a trivial finding -- the anticoagulation-reversal "
            "decision needs to weigh bleeding risk against thrombotic risk from stopping it, which is a real "
            "clinical tension, not a straightforward call.",
        ],
        patient_state_templates=[
            "Active GI bleeding on anticoagulation: hemoglobin {hgb_now} (down from {hgb_start}), melena present; "
            "recommend GI consultation, transfusion threshold assessment, and multidisciplinary discussion of "
            "anticoagulation reversal given {anticoag_reason}.",
        ],
        forecast_vars=["hemoglobin_6h"],
        forecast_ranges={"hemoglobin_6h": (6.5, 9.0, 0.15, 1)},
        urgency="urgent",
    ),
    Archetype(
        name="post_ictal_reassurance",
        contexts=[
            "A {age}-year-old {gender} with no prior psychiatric or seizure history, brought in after a witnessed "
            "seizure, now postictal but {postictal_state}.",
        ],
        think_templates=[
            "Postictal state with {postictal_state} after a single witnessed seizure in someone with no prior "
            "history is expected and typically self-resolving -- the priority is a standard first-seizure workup "
            "(labs, possible imaging) rather than assuming an ongoing emergency at this point.",
        ],
        patient_state_templates=[
            "Postictal after a first witnessed seizure, currently {postictal_state}; recommend standard first-"
            "seizure workup and continued observation as mental status clears.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="moderate",
    ),
    Archetype(
        name="hf_copd_overlap_decompensation",
        contexts=[
            "A {age}-year-old {gender} with heart failure with reduced ejection fraction, admitted for a COPD "
            "exacerbation, has gained {weight_gain}kg and has new bilateral leg edema.",
        ],
        think_templates=[
            "Weight gain of {weight_gain}kg with new bilateral edema in someone with known HFrEF, even in the "
            "context of a COPD admission, points toward volume overload from heart failure decompensation, not "
            "just the pulmonary process -- both need addressing, since treating one without the other risks missing "
            "the actual driver of symptoms.",
        ],
        patient_state_templates=[
            "Likely combined HF decompensation and COPD exacerbation: {weight_gain}kg weight gain, new bilateral "
            "edema; recommend diuresis alongside COPD management, with close respiratory and volume status "
            "monitoring given the overlap.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="moderate",
    ),
    Archetype(
        name="post_thyroidectomy_hypocalcemia",
        contexts=[
            "A {age}-year-old {gender} post-thyroidectomy, {hours}h post-op, reports new tingling around the mouth "
            "and in the fingers, with a calcium of {ca_now}.",
        ],
        think_templates=[
            "Perioral and finger tingling with calcium at {ca_now} post-thyroidectomy is a classic early sign of "
            "post-surgical hypoparathyroidism/hypocalcemia -- this needs calcium repletion and close monitoring for "
            "progression to more serious neuromuscular irritability (Chvostek/Trousseau signs, tetany) before it "
            "gets there, not just symptomatic reassurance.",
        ],
        patient_state_templates=[
            "Post-thyroidectomy hypocalcemia: calcium {ca_now}, new perioral/finger paresthesias; recommend oral or "
            "IV calcium repletion per protocol and monitoring for progression to tetany.",
        ],
        forecast_vars=["calcium_6h"],
        forecast_ranges={"calcium_6h": (7.2, 8.3, 0.06, 1)},
        urgency="moderate",
    ),
    Archetype(
        name="febrile_neutropenia",
        contexts=[
            "A {age}-year-old {gender} with febrile neutropenia (ANC {anc}) post-chemotherapy has a temperature of "
            "{temp}C and new rigors.",
        ],
        think_templates=[
            "Fever ({temp}C) with rigors in someone with ANC {anc} post-chemotherapy is a true oncologic emergency -- "
            "the mortality risk from untreated neutropenic sepsis is high enough that broad-spectrum antibiotics "
            "need to start within the hour, before culture results, not after.",
        ],
        patient_state_templates=[
            "Febrile neutropenia (ANC {anc}, temp {temp}C) with rigors -- oncologic emergency; recommend immediate "
            "blood cultures followed by broad-spectrum empiric antibiotics without delay, per neutropenic fever "
            "protocol.",
        ],
        forecast_vars=["temperature_6h"],
        forecast_ranges={"temperature_6h": (38.5, 40.2, 0.03, 1)},
        urgency="urgent",
    ),
    Archetype(
        name="postpartum_thyroiditis",
        contexts=[
            "A {age}-year-old woman, {weeks} weeks postpartum, presents with new palpitations, heat intolerance, "
            "and unintentional weight loss, with a heart rate of {hr}.",
        ],
        think_templates=[
            "Palpitations, heat intolerance, and weight loss with HR {hr} at {weeks} weeks postpartum is a classic "
            "presentation for postpartum thyroiditis (hyperthyroid phase) -- this is usually self-limited but "
            "warrants thyroid function testing to confirm and guide symptomatic management, and to distinguish it "
            "from other causes of postpartum tachycardia.",
        ],
        patient_state_templates=[
            "Likely postpartum thyroiditis, hyperthyroid phase: HR {hr}, palpitations, heat intolerance, weight "
            "loss; recommend TSH/free T4 testing and symptomatic beta-blockade if needed, with follow-up given the "
            "typical biphasic course.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="moderate",
    ),
    Archetype(
        name="aortic_stenosis_presyncope",
        contexts=[
            "A {age}-year-old {gender} with severe aortic stenosis, admitted for elective {procedure} soon, has new "
            "exertional dizziness and a brief episode of near-syncope.",
        ],
        think_templates=[
            "New exertional dizziness and near-syncope in someone with known SEVERE aortic stenosis is a genuine "
            "warning sign, not incidental -- syncope in severe AS carries real risk of sudden cardiac death, and "
            "this should prompt reassessment of the timeline for intervention rather than waiting for the "
            "originally scheduled date if symptoms are progressing.",
        ],
        patient_state_templates=[
            "New exertional near-syncope in severe aortic stenosis -- concerning symptom given known risk of sudden "
            "cardiac death in symptomatic severe AS; recommend urgent cardiology reassessment of intervention "
            "timing.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="urgent",
    ),
    Archetype(
        name="crohns_flare_improving",
        contexts=[
            "A {age}-year-old {gender} with Crohn's disease, admitted for a flare, has been on IV steroids for "
            "{steroid_days} days with improving stool frequency (from {stools_start} to {stools_now} per day).",
        ],
        think_templates=[
            "Stool frequency improving from {stools_start} to {stools_now} per day after {steroid_days} days of IV "
            "steroids is a genuinely positive response -- this supports continuing the current steroid course with "
            "a plan toward transition to oral steroids and outpatient maintenance therapy if the trend continues.",
        ],
        patient_state_templates=[
            "Crohn's flare responding to IV steroids: stool frequency improving ({stools_start} -> {stools_now}/day) "
            "after {steroid_days} days; plan toward oral steroid transition if trend continues.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="reassuring",
    ),
    Archetype(
        name="pneumothorax_air_leak",
        contexts=[
            "A {age}-year-old {gender} post-{surgery}, day {days}, has a chest tube with a new, sudden increase in "
            "air leak and subcutaneous emphysema.",
        ],
        think_templates=[
            "A new, sudden increase in chest tube air leak with subcutaneous emphysema suggests a new or worsening "
            "air leak from the lung parenchyma or airway -- this needs urgent surgical/pulmonary reassessment for "
            "possible tube malposition, new pneumothorax, or bronchopleural fistula, not just continued observation "
            "on the current management.",
        ],
        patient_state_templates=[
            "New significant air leak with subcutaneous emphysema post-{surgery} -- concerning for evolving "
            "pleural/airway complication; recommend urgent chest imaging and surgical/pulmonary reassessment of "
            "chest tube management.",
        ],
        forecast_vars=[],
        forecast_ranges={},
        urgency="urgent",
    ),
]


def _fmt_forecast_line(var: str, rng: random.Random, lo: float, hi: float, half_width_frac: float, decimals: int) -> str:
    point = rng.uniform(lo, hi)
    half_width = max(point * half_width_frac, 0.05)
    low, high = point - half_width, point + half_width
    fmt = f"%.{decimals}f"
    return f"{var}: {fmt % point} [{fmt % low}-{fmt % high}]"


def _sample_archetype_vars(archetype: Archetype, rng: random.Random) -> dict:
    """Samples every placeholder value this archetype's templates might
    reference. Not every archetype uses every key -- str.format_map with a
    defaultdict-like fallback below tolerates unused keys."""
    demo = _sample_demographics(rng)
    v = dict(demo)
    v.update({
        "hours": rng.choice([4, 6, 8, 12, 18, 24]),
        "days": rng.choice([1, 2, 3, 4, 5]),
        "trend_hours": rng.choice([2, 3, 4, 6]),
        "surgery": rng.choice(["bowel resection", "hip replacement", "total knee replacement", "CABG",
                                "cholecystectomy", "appendectomy", "hysterectomy", "spinal fusion", "lung resection"]),
        "hr": rng.randint(105, 130),
        "map": rng.randint(52, 64),
        "sbp": rng.randint(75, 92),
        "lac_start": round(rng.uniform(1.0, 1.8), 1),
        "lac_now": round(rng.uniform(2.2, 3.4), 1),
        "dm_type": rng.choice(["1", "2"]),
        "glucose": rng.randint(180, 340),
        "ph": round(rng.uniform(7.15, 7.33), 2),
        "ph_interpretation": rng.choice(["still acidotic but improving from the initial value",
                                          "concerningly low and needs close monitoring",
                                          "trending toward normal with current treatment"]),
        "improving_or_not": rng.choice(["she reports feeling much better than on admission",
                                         "she remains nauseated and fatigued",
                                         "mental status is clearing appropriately"]),
        "resp_dx": rng.choice(["pneumonia", "COPD exacerbation", "aspiration pneumonitis", "viral bronchiolitis"]),
        "o2_start": rng.choice(["2L nasal cannula", "room air", "3L nasal cannula", "4L nasal cannula"]),
        "o2_now": rng.choice(["6L nasal cannula", "10L high-flow", "non-rebreather", "8L nasal cannula"]),
        "spo2": rng.randint(84, 90),
        "temp": round(rng.uniform(36.9, 39.3), 1),
        "wbc_start": round(rng.uniform(7.0, 10.0), 1),
        "wbc_now": round(rng.uniform(14.0, 18.5), 1),
        "unit": rng.choice(["ICU", "surgical ward", "step-down unit", "medical floor"]),
        "k_now": round(rng.uniform(5.8, 6.8), 1),
        "k_start": round(rng.uniform(3.8, 4.5), 1),
        "ecg_finding": rng.choice(["new peaked T waves", "a widening QRS", "loss of P waves and peaked T waves"]),
        "delivery_type": rng.choice(["vaginal", "cesarean"]),
        "inr_start": round(rng.uniform(1.1, 1.4), 1),
        "inr_now": round(rng.uniform(1.7, 2.3), 1),
        "mental_status": rng.choice(["worsening confusion", "new asterixis", "increasing daytime somnolence"]),
        "chronic_dx": rng.choice(["asthma", "COPD", "heart failure"]),
        "symptom": rng.choice(["wheezing", "shortness of breath", "productive cough"]),
        "cardiac_hx": rng.choice(["CAD", "a prior MI", "a prior PCI with stent placement"]),
        "trop_now": round(rng.uniform(0.8, 3.2), 2),
        "bipap_hours": rng.choice([4, 6, 8, 12]),
        "pco2_start": rng.randint(72, 85),
        "pco2_now": rng.randint(56, 66),
        "ckd_stage": rng.choice(["3", "3b", "4"]),
        "infection_dx": rng.choice(["community-acquired pneumonia", "a urinary tract infection", "cellulitis"]),
        "cr_start": round(rng.uniform(1.2, 1.6), 1),
        "cr_now": round(rng.uniform(2.0, 2.7), 1),
        "sleep_wake": rng.choice(["has a reversed sleep-wake cycle", "is more agitated at night",
                                   "has intermittent periods of clear lucidity"]),
        "pe_risk": rng.choice(["a prior pulmonary embolism", "a known clotting disorder",
                                "prolonged immobility and recent long-haul travel"]),
        "anticoag_reason": rng.choice(["a mechanical heart valve", "atrial fibrillation", "a prior DVT"]),
        "hgb_start": round(rng.uniform(11.5, 13.5), 1),
        "hgb_now": round(rng.uniform(7.5, 9.5), 1),
        "postictal_state": rng.choice(["appropriate and oriented", "still mildly confused",
                                        "gradually reorienting"]),
        "weight_gain": rng.randint(2, 6),
        "ca_now": round(rng.uniform(7.0, 8.2), 1),
        "anc": rng.randint(100, 480),
        "weeks": rng.choice([2, 4, 6, 8, 10]),
        "procedure": rng.choice(["TAVR", "surgical aortic valve replacement"]),
        "steroid_days": rng.choice([2, 3, 4]),
        "stools_start": rng.randint(8, 12),
        "stools_now": rng.randint(2, 5),
    })
    return v


def _safe_format(template: str, values: dict) -> str:
    class _Default(dict):
        def __missing__(self, key):
            return f"{{{key}}}"
    return template.format_map(_Default(**values))


def generate_archetype_vignettes(n: int, seed: int = 1234) -> list[dict]:
    """Generates n new, independently-sampled vignettes spread across all
    archetypes (round-robin with shuffling, so a large n gets roughly even
    coverage rather than exhausting early archetypes first)."""
    rng = random.Random(seed)
    out = []
    archetype_cycle = list(ARCHETYPES)
    idx = 0
    for i in range(n):
        if idx % len(archetype_cycle) == 0:
            rng.shuffle(archetype_cycle)
        archetype = archetype_cycle[idx % len(archetype_cycle)]
        idx += 1

        values = _sample_archetype_vars(archetype, rng)
        context = _safe_format(rng.choice(archetype.contexts), values)
        think = _safe_format(rng.choice(archetype.think_templates), values)
        patient_state = _safe_format(rng.choice(archetype.patient_state_templates), values)

        if archetype.forecast_vars:
            lines = []
            for var in archetype.forecast_vars:
                lo, hi, half_width_frac, decimals = archetype.forecast_ranges[var]
                lines.append(_fmt_forecast_line(var, rng, lo, hi, half_width_frac, decimals))
            forecast = "\n".join(lines)
        else:
            forecast = "not applicable"

        recipient_type = rng.choices(["clinician", "patient", "family"], weights=[0.70, 0.15, 0.15])[0]
        framing_template = rng.choice(_RECIPIENT_FRAMINGS[recipient_type])
        user_belief = _safe_format(framing_template, {"role": rng.choice(_ROLES)})

        out.append({
            "prompt": context,
            "think": think,
            "patient_state": patient_state,
            "forecast": forecast,
            "user_belief": user_belief,
            "recipient_type": recipient_type,  # extra metadata field; format_synthetic_example ignores unknown keys
            "_archetype": archetype.name,       # extra metadata field, useful for auditing distribution
        })
    return out
