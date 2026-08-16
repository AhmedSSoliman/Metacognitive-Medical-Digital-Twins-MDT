"""
training/sft_formatting.py

Pure string-formatting logic for Phase 1 SFT training examples.

PORTING NOTE (2026-08-12): ported byte-identical (apart from this header and
one import rewrite) from the source repo's training/sft_formatting.py
(../Agentic-DT_V1-July/training/sft_formatting.py). The only code change is
`from models.stream_parsing import STREAM_SYSTEM_PROMPT` ->
`from core.schema import STREAM_SYSTEM_PROMPT`.

JUDGMENT CALL, AND A CORRECTION TO AN EARLIER PORT DECISION: these two
functions were first merged into the top of training/sft.py, above its
`import unsloth` line, on the reasoning that they are SFT-specific and belong
with the pipeline they serve. That was wrong, and the test suite caught it:
Python executes a module top-to-bottom, so `from training.sft import
format_reasoning_example` still runs the `import unsloth` line and therefore
still requires a GPU. Merging them silently destroyed the exact property this
file was created for -- that these pure-string functions stay unit-testable
on a login node or in CI with no ML environment at all.

They are therefore kept in their own module, exactly as the source repo had
them. training/sft.py imports them from here, unchanged in behavior.

WHY THIS IS NOT IN core/: they are not general stream utilities. They encode
the Phase 1 SFT training-example TEMPLATE specifically -- which dataset field
maps to which stream, that general-reasoning examples get "not applicable"
for <forecast>, and the APPEND_ANSWER_TO_THINK diagnostic toggle that exists
purely to A/B a suspected SFT data bug. That is meaningful only to SFT data
prep, not to parsing, rewards, hypergraphs, cohorts, or tools. training/ is
permitted to hold non-heavy-dependency files; the boundary rule is that core/
must not need torch, not that everything torch-free must live in core/.

ORIGINAL training/sft_formatting.py MODULE DOCSTRING:

    Pure string-formatting logic for Phase 1 SFT training examples, split out
    of training/sft_trainer.py for the same reason models/stream_parsing.py was
    split out of models/multi_stream.py: sft_trainer.py's first import is
    `import unsloth`, which requires a GPU just to import the module at all --
    meaning these two formatting functions, despite being pure string logic
    with zero GPU/model dependency, were previously untestable on a non-GPU
    machine (e.g. a login node, or CI). Living here instead, importing only
    models.stream_parsing (itself dependency-free), makes them directly
    unit-testable without any ML environment installed.

    training/sft_trainer.py imports and uses these; nothing about their
    behavior changed by moving them here.
"""

from __future__ import annotations

import os

from core.schema import STREAM_SYSTEM_PROMPT


# Diagnostic toggle, default ON (the real fix): set
# SFT_APPEND_ANSWER_TO_THINK=0 to disable appending the dataset's real
# answer to <think> and revert to the original (bug) behavior. Added to
# isolate whether this specific change is the cause of a real generation-
# termination regression found in a checkpoint trained with it ON (see
# README bug log): all 6 of 6 tested (prompt x decoding-config)
# combinations failed to stop cleanly after </user_belief>, including under
# sampling and repetition-penalty, ruling out a decoding-config explanation.
# Direct evidence pointing at this specific change: the model was observed
# writing a SECOND, free-floating "Answer: ..." immediately after
# </user_belief> too, suggesting it learned "Answer:" as a loosely-scoped
# continuation cue rather than something strictly bound inside <think>.
# This is a short, isolated diagnostic retrain (1 epoch, not the full 3) to
# test that hypothesis before committing to a full retrain either way.
APPEND_ANSWER_TO_THINK = os.environ.get("SFT_APPEND_ANSWER_TO_THINK", "1") == "1"


def format_reasoning_example(ex: dict) -> dict:
    """Formats one tier-one general-medical-reasoning example (e.g. from
    medical-o1-reasoning-SFT) into this project's four-stream training text.

    Adapt field names to the actual dataset schema; this assumes
    'Question'/'Complex_CoT'/'Response' style fields -- confirm exact
    column names against the dataset card before running.
    """
    question = ex.get("Question") or ex.get("question", "")
    cot = ex.get("Complex_CoT") or ex.get("reasoning", "")
    answer = ex.get("Response") or ex.get("answer", "")
    # `answer` was previously extracted here but never used anywhere in the
    # template below -- the model was trained on ~92% of Phase 1's data
    # (every non-ICU-vignette example) to end <think> without ever stating a
    # concluding answer, while <user_belief> was hardcoded to a fixed,
    # uninformative placeholder rather than any real content. Confirmed via
    # a real generation from a trained checkpoint: asked a genuine clinical
    # question, the model produced fully correct reasoning inside <think>
    # but its displayed "final answer" (<user_belief>) was just that same
    # fixed placeholder string, verbatim -- exactly what this data bug would
    # predict. Chain-of-thought naturally concludes with a stated answer, so
    # appending the dataset's own answer to the end of the reasoning trace
    # keeps it complete and gives the model something substantive to
    # actually learn to produce, rather than discarding real training signal.
    think_content = f"{cot}\n\nAnswer: {answer}" if (answer and APPEND_ANSWER_TO_THINK) else cot
    # Stream order MUST match STREAM_TAGS in models/stream_parsing.py exactly
    # (think, patient_state, forecast, user_belief) -- forecast is "not
    # applicable" here since general medical-reasoning questions have no ICU
    # deterioration forecast target.
    text = (
        f"{STREAM_SYSTEM_PROMPT}\n\n### Input\n{question}\n\n### Response\n"
        f"<think>{think_content}</think>"
        f"<patient_state>Not applicable for this general-reasoning example.</patient_state>"
        f"<forecast>not applicable</forecast>"
        f"<user_belief>Assume a clinician-level reader for this general reasoning task.</user_belief>"
    )
    return {"text": text}


def format_synthetic_example(ex: dict) -> dict:
    """Formats one hand-written, stream-format-annotated ICU vignette
    (data/synthetic/stream_format_vignettes.jsonl) into training text.

    `forecast` is optional in the vignette JSONL for backward compatibility
    with vignette files written before the forecast stream was added --
    defaults to "not applicable" if absent. To actually teach the model the
    structured forecast format, vignettes need this field populated with
    real "VAR_Nh: value [low-high]" lines.
    """
    forecast_content = ex.get("forecast", "not applicable")
    text = (
        f"{STREAM_SYSTEM_PROMPT}\n\n### Input\n{ex['prompt']}\n\n### Response\n"
        f"<think>{ex['think']}</think>"
        f"<patient_state>{ex['patient_state']}</patient_state>"
        f"<forecast>{forecast_content}</forecast>"
        f"<user_belief>{ex['user_belief']}</user_belief>"
    )
    return {"text": text}
