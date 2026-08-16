"""
tests/test_training/test_grpo_math.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/test_grpo_math.py.
Only imports and path references were rewritten for the new package layout;
no test function was added, removed, renamed, or had its assertions changed.

Formalizes manual verification of GRPO's group-relative-advantage
computation: zero-centering, correct scaling, and that the highest-reward
completion in a group always receives the highest advantage (required for
the policy-gradient sign to point the right direction).

Imports from training.grpo_utils, which has no torch-heavy dependency beyond
torch itself for sequence_logprob -- but compute_group_relative_advantage is
pure numpy, so this file is marked requires_torch only because the module
import line also pulls in torch (for the other function in the same file).
If torch isn't installed, this test file is skipped by
`pytest -m "not requires_torch"`.
"""

import numpy as np
import pytest

pytest.importorskip("torch", reason="training.grpo_utils imports torch at module level")

import sys
from pathlib import Path
from training.grpo_utils import compute_group_relative_advantage

pytestmark = pytest.mark.requires_torch


def test_advantage_is_zero_centered():
    rewards = np.array([0.2, 0.5, 0.8, 0.3, 0.9, 0.1, 0.6, 0.4])
    advantages = compute_group_relative_advantage(rewards)
    assert abs(advantages.mean()) < 1e-6


def test_highest_reward_gets_highest_advantage():
    rewards = np.array([0.2, 0.5, 0.8, 0.3, 0.9, 0.1, 0.6, 0.4])
    advantages = compute_group_relative_advantage(rewards)
    assert np.argmax(rewards) == np.argmax(advantages)


def test_lowest_reward_gets_lowest_advantage():
    rewards = np.array([0.2, 0.5, 0.8, 0.3, 0.9, 0.1, 0.6, 0.4])
    advantages = compute_group_relative_advantage(rewards)
    assert np.argmin(rewards) == np.argmin(advantages)


def test_degenerate_all_identical_rewards_does_not_crash():
    # Every completion in the group scored identically -- std would be exactly
    # 0 without the epsilon guard, causing a division-by-zero. Verifies the
    # epsilon in compute_group_relative_advantage prevents this.
    rewards = np.array([0.5, 0.5, 0.5, 0.5])
    advantages = compute_group_relative_advantage(rewards)
    assert np.all(np.isfinite(advantages))
    assert np.allclose(advantages, 0.0, atol=1e-3)
