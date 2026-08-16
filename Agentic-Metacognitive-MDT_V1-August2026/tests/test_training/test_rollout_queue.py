"""
tests/test_training/test_rollout_queue.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_rollout_queue.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Tests RolloutQueue's file-based put/get_batch mechanics in isolation --
these are pure file I/O and JSON serialization, no torch/model dependency,
so this file has no requires_torch marker and runs anywhere.
"""

import sys
import tempfile
import os
from pathlib import Path

from training.rollout import RolloutQueue, RolloutResult


def _make_dummy_result(worker_id=0):
    return RolloutResult(
        prompt_id="p1", completion="dummy", reward_components={"total": 0.7},
        tool_calls_made=0, generation_latency_ms=120.0, worker_id=worker_id,
    )


def test_put_and_get_batch_roundtrip():
    tmp_path = tempfile.mktemp(suffix=".jsonl")
    try:
        rq = RolloutQueue(tmp_path)
        for _ in range(10):
            rq.put(_make_dummy_result())

        batch = rq.get_batch(max_items=5)
        assert len(batch) == 5

        remaining = rq.get_batch(max_items=100)
        assert len(remaining) == 5
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_get_batch_consumes_items_not_just_peeks():
    """Verifies get_batch actually REMOVES consumed items from the queue
    file rather than leaving them for a second read to pick up again --
    a queue that doesn't consume would silently double-process results."""
    tmp_path = tempfile.mktemp(suffix=".jsonl")
    try:
        rq = RolloutQueue(tmp_path)
        for _ in range(3):
            rq.put(_make_dummy_result())

        first_read = rq.get_batch(max_items=100)
        second_read = rq.get_batch(max_items=100, timeout_s=1.0)
        assert len(first_read) == 3
        assert len(second_read) == 0
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_get_batch_on_nonexistent_queue_file_returns_empty_not_error():
    tmp_path = tempfile.mktemp(suffix=".jsonl")
    rq = RolloutQueue(tmp_path)
    result = rq.get_batch(max_items=5, timeout_s=1.0)
    assert result == []
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_result_fields_survive_json_roundtrip():
    tmp_path = tempfile.mktemp(suffix=".jsonl")
    try:
        rq = RolloutQueue(tmp_path)
        original = RolloutResult(
            prompt_id="step5_idx12", completion="<think>test</think>",
            reward_components={"total": 0.85, "R_format": 1.0}, tool_calls_made=2,
            generation_latency_ms=456.7, worker_id=1,
        )
        rq.put(original)
        result = rq.get_batch(max_items=1)[0]
        assert result["prompt_id"] == "step5_idx12"
        assert result["reward_components"]["total"] == 0.85
        assert result["tool_calls_made"] == 2
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
