"""
tests/test_training/test_generate_retry.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_generate_retry.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Tests MultiStreamModel.generate()'s retry-on-malformed logic in isolation,
using a mocked underlying model/tokenizer (no real GPU model needed -- this
tests the ORCHESTRATION logic: does it retry the right number of times,
does it stop early on success, does it give up gracefully). Added as a
practical mitigation for a real, confirmed generation-termination defect
(see README bug log / training/backbone.py's matching docstring).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock


WELL_FORMED = "<think>t</think><patient_state>p</patient_state><forecast>not applicable</forecast><user_belief>u</user_belief>"
MALFORMED = "<think>t</think><patient_state>p</patient_state>"  # missing forecast/user_belief


def _make_model_with_responses(responses):
    """Builds a MultiStreamModel-like object whose _generate_once returns
    the given responses in sequence, one per call."""
    from training.backbone import MultiStreamModel
    obj = MagicMock()
    obj.cfg = MagicMock(max_seq_len=4096)
    obj.tokenizer = MagicMock()
    fake_inputs = MagicMock()
    fake_inputs.to.return_value = fake_inputs
    obj.tokenizer.return_value = fake_inputs
    obj.format_prompt = MagicMock(return_value="formatted prompt")
    call_iter = iter(responses)
    obj._generate_once = MagicMock(side_effect=lambda *a, **k: next(call_iter))
    # Bind the REAL generate() method (unbound function) to this mock object
    # so we're testing the actual retry logic, not a re-implementation of it.
    obj.generate = MultiStreamModel.generate.__get__(obj, MultiStreamModel)
    return obj


def test_succeeds_on_first_attempt_no_retry_needed():
    model = _make_model_with_responses([[WELL_FORMED]])
    result = model.generate("prompt")
    assert result == [WELL_FORMED]
    assert model._generate_once.call_count == 1


def test_retries_then_succeeds():
    model = _make_model_with_responses([[MALFORMED], [MALFORMED], [WELL_FORMED]])
    result = model.generate("prompt", max_retries_if_malformed=2)
    assert result == [WELL_FORMED]
    assert model._generate_once.call_count == 3


def test_gives_up_after_max_retries_returns_last_attempt():
    model = _make_model_with_responses([[MALFORMED], [MALFORMED], [MALFORMED]])
    result = model.generate("prompt", max_retries_if_malformed=2)
    assert result == [MALFORMED]
    assert model._generate_once.call_count == 3  # 1 initial + 2 retries, then gives up


def test_zero_retries_means_single_attempt_only():
    model = _make_model_with_responses([[MALFORMED]])
    result = model.generate("prompt", max_retries_if_malformed=0)
    assert result == [MALFORMED]
    assert model._generate_once.call_count == 1


def test_retry_logic_not_applied_when_num_return_sequences_greater_than_one():
    """GRPO-style group sampling (num_return_sequences > 1) must NOT be
    retried -- see the docstring in training/backbone.py for why."""
    group_result = [MALFORMED, MALFORMED, WELL_FORMED, MALFORMED]
    model = _make_model_with_responses([group_result])
    result = model.generate("prompt", num_return_sequences=4, max_retries_if_malformed=2)
    assert result == group_result
    assert model._generate_once.call_count == 1  # no retry attempted
