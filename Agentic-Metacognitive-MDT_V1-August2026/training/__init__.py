"""
training

Requires torch, TRL, and Unsloth. Every module here imports at least one of
them at module level (training/sft.py and training/grpo.py import unsloth,
which raises without a GPU) -- this is the deliberate other side of the
core/ dependency boundary.

  sft.py             Aim 1: Phase 1 supervised fine-tuning (+ its example
                     formatters, see that file's placement note).
  grpo.py            Aim 2: Phase 2 GRPO training loop, on TRL's GRPOTrainer.
  grpo_utils.py      GRPO math used by Phase 4's manual loop (NOT by grpo.py).
  rollout.py         Aim 4: rollout-worker pool + the scaled-training driver.
  backbone.py        Model loading, LoRA config, generation, backbone selection.
  context_pruning.py Placeholder -- not implemented; documents where
                     truncation currently happens instead.
"""
