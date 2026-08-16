"""
training/grpo_utils.py

PORTING NOTE (2026-08-12): ported byte-identical from the source repo's
training/grpo_utils.py (../Agentic-DT_V1-July/training/grpo_utils.py). It has
no intra-project imports, so nothing needed rewriting.

JUDGMENT CALL -- KEPT AS A SEPARATE FILE rather than folded into
training/grpo.py, for the exact reason the source file's own docstring gives
below: these three names (PromptDataset, compute_group_relative_advantage,
sequence_logprob) are consumed by PHASE 4 (training/rollout.py's driver
loop), NOT by Phase 2's TRL-based training/grpo.py, which needs none of them.
Folding them into grpo.py would recreate the precise coupling that already
broke once -- Phase 4 importing GRPO mechanics from Phase 2's trainer, then
silently breaking with an ImportError when Phase 2 was refactored onto TRL.
The target tree does not list this filename, but it explicitly allows extra
same-directory helper modules; this is one.

ORIGINAL training/grpo_utils.py MODULE DOCSTRING:

    Shared GRPO utility functions used by Phase 4's custom, distributed rollout
    scaling approach (scripts/run_phase4_scaled_training.py).

    WHY THIS FILE EXISTS SEPARATELY FROM training/grpo_trainer.py: Phase 2's
    grpo_trainer.py was refactored to use TRL's own GRPOTrainer directly (see
    that file's module docstring for why), which internally handles group
    sampling, advantage computation, and sequence log-probability computation
    -- so those hand-rolled implementations no longer live there. Phase 4's
    scaling approach is architecturally different: it runs its own pool of
    rollout worker PROCESSES (agents/rollout_service.py) that generate and score
    completions in parallel, feeding results back to a single training process
    over a shared queue. TRL's GRPOTrainer does not expose a plug-in point for
    that kind of external, multi-process rollout pattern, so Phase 4 still needs
    its own advantage computation and sequence log-probability code to perform
    policy-gradient updates directly. Rather than duplicating this code (or,
    worse, letting Phase 4 silently break when Phase 2 was refactored -- which
    is exactly what happened until this file was added, since
    run_phase4_scaled_training.py originally imported these three names FROM
    grpo_trainer.py, which no longer define them after the TRL refactor), they
    live here as a single shared, tested location.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Wraps a JSONL/parquet file of prompts + reference labels needed for
    reward computation (reference_patient_state, recipient_type,
    must_mention_facts). This is the output format produced by
    data/build_grpo_prompts.py.
    """

    def __init__(self, data_path: str):
        if data_path.endswith(".parquet"):
            self.df = pd.read_parquet(data_path)
        else:
            self.df = pd.read_json(data_path, lines=True)
        required_cols = {"prompt", "reference_patient_state", "recipient_type"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Prompt dataset missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "prompt": row["prompt"],
            "reference_patient_state": row["reference_patient_state"],
            # .get() with a default handles rows/datasets where these
            # optional columns are absent, rather than raising a KeyError --
            # recipient_type defaults to "clinician" (the most common case
            # per core/cohort/grpo_prompts.py's sampling distribution), and
            # must_mention_facts defaults to an empty list (meaning "nothing
            # specific to check for retention on this example").
            "recipient_type": row.get("recipient_type", "clinician"),
            "must_mention_facts": row.get("must_mention_facts", []) or [],
            # Both optional (older cached prompt datasets built before
            # core/cohort/grpo_prompts.py started carrying these through
            # won't have them) -- only consumed by Phase 4's agentic
            # tool-use path (core/tools/dispatch.py's get_recent_labs), which
            # degrades to its "no data source configured" placeholder
            # response rather than erroring when either is missing.
            "hadm_id": row.get("hadm_id", None),
            "prediction_time": (str(row["prediction_time"]) if "prediction_time" in row.index
                                 and pd.notna(row.get("prediction_time")) else None),
        }


def compute_group_relative_advantage(rewards: np.ndarray) -> np.ndarray:
    """GRPO's defining mechanic: the advantage of each completion is just
    its reward normalized against the MEAN AND STD OF ITS OWN GROUP (the G
    completions sampled for the same prompt) -- not a global baseline, and
    not a separately trained value/critic model. This is what makes GRPO
    cheaper to run than PPO-style RLHF: no second network to train and keep
    in sync with the policy.

    Verified empirically (see the project's development history) that this
    correctly zero-centers each group and that the highest-reward completion
    in a group always receives the highest advantage, as it must for the
    policy-gradient sign to point in the right direction.
    """
    mean = rewards.mean()
    std = rewards.std() + 1e-6  # epsilon guards against a degenerate all-identical-reward group (std=0)
    return (rewards - mean) / std


def sequence_logprob(model, tokenizer, prompt: str, completion: str, device) -> torch.Tensor:
    """Sum log-probability of the completion tokens under the current policy,
    conditioned on the prompt. Used both for the policy-gradient term and the
    KL-to-reference penalty in Phase 4's manual training loop.

    IMPORTANT: this tokenizes `prompt` and `completion` SEPARATELY and then
    concatenates the resulting token ID tensors directly -- it deliberately
    does NOT tokenize `prompt + completion` as one string and try to recover
    the boundary from `len(tokenizer(prompt).input_ids)`. That approach is a
    real, well-documented pitfall this project's own code hit and fixed
    during development: BPE-style tokenizers can merge tokens differently
    across the prompt/completion boundary depending on context (e.g.
    trailing/leading whitespace merging into an adjacent token), so a
    separately-computed prompt length can silently misalign with where the
    completion actually starts inside the jointly-tokenized sequence --
    corrupting the reward signal without raising any error at all, which is
    what makes this bug class dangerous: it fails silently, not loudly.
    Concatenating token IDs directly guarantees the boundary is exactly
    where intended, at the cost of a negligible loss of cross-boundary BPE
    merge optimality, which is the standard accepted tradeoff in RL-for-LLM
    training code generally.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)

    # Gradient tracking follows the model's own .training flag: when scoring
    # completions under the FROZEN reference model (ref_model.eval(), used
    # only for the KL penalty), no_grad saves memory since those gradients
    # are never used; when scoring under the live POLICY model during an
    # actual update step, gradients must flow through for backprop to work.
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(full_ids)
    logits = outputs.logits[:, :-1, :]   # shift by one: logits at position i predict token i+1
    targets = full_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

    # The boundary is now known exactly: prompt_ids.shape[1] tokens, no
    # re-tokenization involved, so this index is trustworthy (see the
    # docstring above for why this matters). The -1 accounts for the
    # logits/targets shift-by-one applied above.
    completion_start = prompt_ids.shape[1] - 1
    completion_logprobs = token_logprobs[:, completion_start:]
    return completion_logprobs.sum()
