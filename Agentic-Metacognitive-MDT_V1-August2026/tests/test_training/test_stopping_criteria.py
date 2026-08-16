"""
tests/test_training/test_stopping_criteria.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_stopping_criteria.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Tests training/backbone.py's _StopOnClosingTag -- added after a real,
confirmed generation-termination regression where two independently
trained Phase 1 checkpoints failed to stop cleanly after </user_belief> in
12/12 tested (prompt x decoding-config) combinations. This is a targeted,
model-independent unit test of the stopping-criteria LOGIC using a
lightweight tokenizer (does not need the real gated MedGemma tokenizer or
a GPU) rather than an end-to-end generation test.
"""

import sys
from pathlib import Path

import pytest
import torch


transformers = pytest.importorskip("transformers")
from transformers import AutoTokenizer
from training.backbone import _StopOnClosingTag


@pytest.fixture(scope="module")
def tokenizer():
    # A small, freely-downloadable tokenizer -- sufficient to test the
    # decode-and-substring-match LOGIC, which does not depend on which
    # specific model's tokenizer is used.
    return AutoTokenizer.from_pretrained("gpt2")


def test_not_done_before_closing_tag(tokenizer):
    text = "<think>reasoning</think><patient_state>s</patient_state><forecast>not applicable</forecast><user_belief>still writing"
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    criterion = _StopOnClosingTag(tokenizer)
    result = criterion(ids, None)
    assert result.tolist() == [False]


def test_done_after_closing_tag(tokenizer):
    text = "<think>reasoning</think><patient_state>s</patient_state><forecast>not applicable</forecast><user_belief>done</user_belief>"
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    criterion = _StopOnClosingTag(tokenizer)
    result = criterion(ids, None)
    assert result.tolist() == [True]


def test_per_sequence_batch_independence(tokenizer):
    """Different sequences in a batch (e.g. num_return_sequences > 1) must
    be evaluated independently -- one finishing early shouldn't affect
    another still-generating sequence's result."""
    done_text = "<user_belief>done</user_belief>"
    not_done_text = "<user_belief>still going"
    # Pad both to the same length for a real batch tensor.
    done_ids = tokenizer(done_text, return_tensors="pt")["input_ids"][0]
    not_done_ids = tokenizer(not_done_text, return_tensors="pt")["input_ids"][0]
    max_len = max(len(done_ids), len(not_done_ids))
    pad_id = tokenizer.eos_token_id or 0

    def pad(ids):
        return torch.cat([ids, torch.full((max_len - len(ids),), pad_id, dtype=ids.dtype)])

    batch = torch.stack([pad(done_ids), pad(not_done_ids)])
    criterion = _StopOnClosingTag(tokenizer)
    result = criterion(batch, None)
    assert result.tolist() == [True, False]


def test_does_not_false_positive_on_unrelated_text(tokenizer):
    text = "<think>The patient's belief system and user preferences vary, but nothing about beliefs is closing yet</think>"
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    criterion = _StopOnClosingTag(tokenizer)
    result = criterion(ids, None)
    assert result.tolist() == [False]
