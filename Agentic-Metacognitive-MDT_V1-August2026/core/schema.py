"""
core/schema.py

Four-Stream architecture SCHEMA/SPEC constants: the stream tag vocabulary,
the system prompt that specifies the output contract, and the constrained
forecast sub-format example.

PORTING NOTE (2026-08-12): this is the constants half of the source repo's
models/stream_parsing.py (../Agentic-DT_V1-July/models/stream_parsing.py).
The parsing FUNCTIONS from that same file now live in core/parsing.py. The
split is purely organizational -- no logic, values, or behavior changed.
The original module docstring is preserved verbatim below because it
documents the real bug that motivated keeping this logic torch-free, which
is exactly the `core/` dependency-boundary principle this whole package
layout is built around.

ORIGINAL models/stream_parsing.py MODULE DOCSTRING:

    Pure-logic stream definitions and parsing for the Four-Stream architecture --
    deliberately kept FREE of any torch/transformers/peft dependency.

    WHY THIS MODULE EXISTS SEPARATELY FROM models/multi_stream.py: it originally
    didn't. STREAM_TAGS, ParsedStreams, parse_streams, and parse_forecast_text
    were all defined directly in multi_stream.py, which also imports torch,
    transformers, and peft at module level for the model-loading classes. That
    meant importing even pure regex-parsing logic -- with no model, no GPU, no
    generation involved at all -- required a full ML environment to be
    installed, which silently defeated the point of writing "torch-independent"
    unit tests for the parsing logic (discovered when tests/test_stream_parsing.py
    failed to even COLLECT with `ModuleNotFoundError: No module named 'torch'`
    in an environment with pytest but no torch). Splitting this out means the
    parsing logic can be tested, imported, and reused (e.g. a lightweight script
    that just re-parses already-generated text from a log file) without needing
    the full training/inference stack installed.

    models/multi_stream.py now imports everything it needs from this module
    rather than redefining it, so there is exactly one definition of the stream
    format and its parser, not two that could drift out of sync.

NOTE ON "LOINC MAPPINGS" (target-tree spec): the target structure listed
"LOINC mappings" as belonging in this file. No LOINC codes exist anywhere in
the source repo (verified by grep for 'loinc' across the whole tree -- the
only hits are in cohort-stratification code, and those are ICD-9/ICD-10
prefixes, not LOINC). What this project actually uses for lab/vital
identification are MIMIC-IV `itemid` integers (see
core/cohort/mimic.py's MimicConfig.vital_itemids / lab_itemids) and the
free-text abnormality patterns in core/hypergraph/verification.py. A real
LOINC mapping layer would belong in core/cohort/terminology.py; see that
file's placeholder note.
"""

from __future__ import annotations

STREAM_TAGS = ["think", "patient_state", "forecast", "user_belief"]

# The forecast stream uses a constrained, parseable sub-format (not free text)
# specifically because free-text numeric generation from LLMs is known to be
# unreliable for precise quantitative forecasting -- constraining the format
# at least makes the numbers machine-parseable and auditable, even though the
# underlying numeric accuracy still depends on the model (see ForecastHead in
# models/multi_stream.py for a statistically more principled alternative/complement).
FORECAST_FORMAT_EXAMPLE = "MAP_6h: 58 [52-64]\nlactate_6h: 3.1 [2.4-3.9]"

STREAM_SYSTEM_PROMPT = (
    "You must structure every response using exactly four tagged sections, "
    "in this order, each opened and closed with matching XML-style tags:\n"
    "<think> ... your internal reasoning ... </think>\n"
    "<patient_state> ... structured summary of the patient's current physiological state ... </patient_state>\n"
    "<forecast> ... structured numeric predictions, one variable per line, in the exact format "
    "'VARIABLE_HORIZONh: VALUE [LOW-HIGH]' (a 95%-interval-style range), e.g.:\n"
    f"{FORECAST_FORMAT_EXAMPLE}\n"
    "... </forecast>\n"
    "<user_belief> ... your estimate of what the reader already knows and how they may be feeling ... </user_belief>\n"
    "All four sections are required in every response, in this exact order. If a forecast is not "
    "applicable to the current question, still include the tag with the single line 'not applicable'."
)
