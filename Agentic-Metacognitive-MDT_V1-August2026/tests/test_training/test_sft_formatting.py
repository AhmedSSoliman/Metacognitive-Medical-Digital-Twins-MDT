"""
tests/test_training/test_sft_formatting.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_sft_formatting.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Tests training/sft.py -- specifically a real data bug found while
testing a trained checkpoint through the Streamlit app: format_reasoning_example
extracted the dataset's real answer into a local variable but never used it
anywhere, so every general-reasoning training example taught the model to
end <think> without a stated conclusion and to always emit an identical,
uninformative <user_belief> placeholder as its "final answer" -- confirmed
via a real generation where the model reproduced that exact placeholder
verbatim for a genuine clinical question. No torch/unsloth dependency --
this is pure string-formatting logic.
"""

import sys
from pathlib import Path

import pytest

# PORT NOTE: imports from training/sft_formatting.py (unchanged name and
# location relative to the source repo). Needs neither torch nor a GPU --
# see that module's header for why it is deliberately kept separate from
# training/sft.py, which does require both.
from training.sft_formatting import format_reasoning_example, format_synthetic_example


def test_format_reasoning_example_appends_real_answer_to_think():
    """Regression test for the real bug: the dataset's answer field must
    actually appear in the training text, not be silently discarded."""
    ex = {
        "Question": "What causes a dry cough with ACE inhibitors?",
        "Complex_CoT": "ACE inhibitors block bradykinin breakdown.",
        "Response": "Bradykinin accumulation irritates the airway.",
    }
    out = format_reasoning_example(ex)
    assert "ACE inhibitors block bradykinin breakdown." in out["text"]
    assert "Bradykinin accumulation irritates the airway." in out["text"]
    # The answer must appear INSIDE the actual <think> response block, not
    # just anywhere in the text (the system-prompt instructions embedded
    # earlier in the text also mention "<think>" as part of the format spec).
    response_section = out["text"].split("### Response\n", 1)[1]
    think_start = response_section.index("<think>")
    think_end = response_section.index("</think>")
    assert think_start < response_section.index("Bradykinin accumulation") < think_end


def test_format_reasoning_example_handles_missing_answer_gracefully():
    """No answer field present (or empty) -- should not crash, and should
    not append a stray 'Answer: ' with nothing after it."""
    ex = {"Question": "A question with no answer field.", "Complex_CoT": "Some reasoning."}
    out = format_reasoning_example(ex)
    assert "Some reasoning." in out["text"]
    assert "Answer:" not in out["text"]


def test_format_reasoning_example_uses_alternate_field_names():
    """Some dataset variants use lowercase 'question'/'reasoning'/'answer'
    instead of the medical-o1-reasoning-SFT 'Question'/'Complex_CoT'/'Response'
    schema -- both must work."""
    ex = {"question": "Alt schema question", "reasoning": "Alt reasoning", "answer": "Alt answer"}
    out = format_reasoning_example(ex)
    assert "Alt schema question" in out["text"]
    assert "Alt reasoning" in out["text"]
    assert "Alt answer" in out["text"]


def test_format_reasoning_example_produces_all_four_streams_in_order():
    ex = {"Question": "Q", "Complex_CoT": "C", "Response": "A"}
    out = format_reasoning_example(ex)
    # Only the actual response section, not the system-prompt instructions
    # (which also mention all four tag names as part of the format spec).
    response_section = out["text"].split("### Response\n", 1)[1]
    for tag in ["<think>", "<patient_state>", "<forecast>", "<user_belief>"]:
        assert tag in response_section
    assert (response_section.index("<think>") < response_section.index("<patient_state>")
            < response_section.index("<forecast>") < response_section.index("<user_belief>"))


def test_format_synthetic_example_uses_real_vignette_fields():
    ex = {
        "prompt": "A patient vignette.",
        "think": "Vignette reasoning.",
        "patient_state": "Tachycardic, hypotensive.",
        "user_belief": "Reader is a bedside nurse.",
        "forecast": "MAP_6h: 58 [52-64]",
    }
    out = format_synthetic_example(ex)
    assert "Vignette reasoning." in out["text"]
    assert "Tachycardic, hypotensive." in out["text"]
    assert "MAP_6h: 58 [52-64]" in out["text"]
    assert "Reader is a bedside nurse." in out["text"]


def test_format_synthetic_example_defaults_missing_forecast_to_not_applicable():
    """Backward compatibility with vignette files written before the
    forecast stream was added."""
    ex = {
        "prompt": "A patient vignette.",
        "think": "Reasoning.",
        "patient_state": "State.",
        "user_belief": "Belief.",
    }
    out = format_synthetic_example(ex)
    assert "<forecast>not applicable</forecast>" in out["text"]
