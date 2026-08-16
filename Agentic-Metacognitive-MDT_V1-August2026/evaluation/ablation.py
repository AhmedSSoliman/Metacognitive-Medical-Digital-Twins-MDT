"""
evaluation/ablation.py

PLACEHOLDER -- there is NO pre-registered ablation runner in this project yet.

Status as of this port (2026-08-12): grepping the source repo
(../Agentic-DT_V1-July/) for 'ablation' returns exactly ONE match, and it is
not a runner, a script, or a flag. It is a single word inside a code comment
in training/sft_trainer.py (now training/sft.py), on the SFTConfig optimizer
choice:

    # NOT paged_adamw_8bit: switched to the standard (non-quantized)
    # optimizer as an ablation, given the raw pre-clip grad_norm
    # observed on real data is extremely unstable (fluctuating across
    # six orders of magnitude step to step, up to 1.4 million in one
    # diagnostic run) -- an 8-bit quantized optimizer's internal
    # moment/variance state is a plausible additional failure point
    # under that kind of input, on top of whatever upstream Gemma3/
    # Unsloth numerical issue produces the instability in the first place.
    optim="adamw_torch",

That is a one-off manual A/B someone ran by hand and then hardcoded the
winner of -- it is not a reproducible ablation harness, and nothing sweeps
it. See training/sft.py, inside main()'s build_sft_config().

Two other things in the codebase are ablation-SHAPED but are also not
pre-registered ablation runners, listed so they are not mistaken for one:
  - training/sft.py's --soft_alignment / --no-soft_alignment flag, which
    toggles the two-stage curriculum against the single-stage baseline. This
    is the closest thing to a real, wired-up ablation switch in the project,
    and scripts/run_sft.sh currently passes --no-soft_alignment.
  - The v2/v3/v4/diag_noanswer Phase 1 retrains documented in the source
    repo's docs/hypothesis_log.md, which varied one factor at a time by
    editing SBATCH scripts. Manual, not automated.

WHAT A REAL RUNNER WOULD NEED: a declared list of conditions (which reward
components are enabled in core/rewards/composite.py's RewardWeights, whether
soft alignment is on, which hypergraph mode), registered BEFORE running --
that is what "pre-registered" means here; the point is to fix the comparison
set in advance so it cannot be chosen post-hoc to favor a result -- plus a
driver that runs evaluation/report.py's run_full_evaluation once per
condition and tabulates the deltas. scripts/run_ablation.sh is the
corresponding SBATCH placeholder.
"""

from __future__ import annotations
