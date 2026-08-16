"""
training/grpo.py

PORTING NOTE (2026-08-12): ported from the source repo's
training/grpo_trainer.py (../Agentic-DT_V1-July/training/grpo_trainer.py).
Only the intra-project import block changed:
  - the model wrapper's re-exported STREAM_SYSTEM_PROMPT/parse_streams ->
    `from core.schema import STREAM_SYSTEM_PROMPT` +
    `from core.parsing import parse_streams` (importing the torch-free
    originals directly rather than via the torch-importing wrapper)
  - `from training.rewards import (...)` -> `from core.rewards import (...)`
  - the hypergraph checkers -> `from core.hypergraph.verification import ...`
  - `from training.sft_trainer import GradientCollapseDetector`
    -> `from training.sft import GradientCollapseDetector`
  - the `sys.path.append(...)` hack is dropped (proper package imports now).
Everything else -- every env-var comment, every job-number bug-log note, every
hyperparameter default -- is byte-identical.

ORIGINAL training/grpo_trainer.py MODULE DOCSTRING:

    Phase 2: Multi-Objective GRPO Alignment -- built on TRL's own `GRPOTrainer`
    rather than a hand-rolled training loop.

    This replaces an earlier hand-rolled implementation. That version worked
    after fixing a real tokenization-boundary bug (see the project README's bug
    log), but reimplemented mechanics (group sampling, sequence log-prob
    computation, KL penalty) that TRL's GRPOTrainer already implements, tests,
    and maintains. Ahmed's own prior repository (AhmedSSoliman/medical-digital-
    twin) already uses TRL's GRPOTrainer successfully with Unsloth, so this
    version follows that same proven integration pattern instead of reinventing
    it, while extending it to the full 4-stream architecture (think,
    patient_state, forecast, user_belief) and 9-component reward set (vs. the
    prior repo's single <think>-tag format and 2-component reward).

    Key integration points adopted directly from that working setup:
      - Unsloth's `FastLanguageModel` + `PatchFastRL("GRPO", FastLanguageModel)`
        for memory-efficient loading and Unsloth/TRL compatibility.
      - `fast_inference=False` and `use_vllm=False` for Python 3.13 / environment
        compatibility (avoids known vLLM integration issues).
      - TRL's reward-function signature: `def reward_func(completions, **kwargs)
        -> list[float]`, where `completions` may be either raw strings or
        conversational message-dicts (`completion[0]["content"]`), and any extra
        dataset columns (e.g. `reference_patient_state`, `recipient_type`) are
        automatically passed through as keyword arguments if present in the
        training dataset.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

# Captured HERE, before any Unsloth-related import or model load, because
# unsloth/models/loader.py's FastLanguageModel.from_pretrained() UNCONDITIONALLY
# resets os.environ["UNSLOTH_FORCE_FLOAT32"] = "0" internally on every call,
# then only re-sets it to "1" itself based on its OWN architecture+dtype
# heuristic (checks the resolved dtype against torch.float16 / bfloat16
# hardware support) -- a heuristic that does NOT fire for our case (bf16
# dtype on bf16-capable hardware), regardless of what this env var was
# explicitly set to beforehand. Confirmed via direct GPU diagnostics (job
# 38030835): the Gemma3 attention patches themselves ARE correctly
# installed using the ORIGINAL value (since `import unsloth` below runs
# BEFORE from_pretrained() ever executes and applies its own reset), but
# any code that re-reads the env var AFTER model loading (as an earlier
# version of this file's generation_kwargs logic did) sees the wrong,
# already-reset value. Reading and freezing it into a module-level
# constant up front avoids this trap entirely.
_UNSLOTH_FORCE_FLOAT32_REQUESTED = os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"

# See training/sft.py's matching comment: Unsloth must be imported
# before trl/transformers/peft/datasets, confirmed via direct GPU
# diagnostics to affect more than performance -- the wrong order (trl
# imported at module top-level, unsloth only imported later inside a
# function) was linked to real training runs silently ending up with 0
# trainable parameters. Tradeoff: unsloth requires a GPU to import at all,
# so this script can no longer be imported on a non-GPU machine -- accepted
# since every real invocation is already a SLURM GPU job.

# Jobs 37534921/37594245 both crashed ~1.5h into rollout generation with
# `torch._dynamo.exc.FailOnRecompileLimitHit: Hard failure due to
# fullgraph=True`. Root cause: UNSLOTH_FORCE_FLOAT32=1 (required for Gemma3
# training stability -- see that env var's own bug log entry) patches
# Gemma3RMSNorm.forward with torch.compile(fullgraph=True)
# (unsloth_zoo/temporary_patches/gemma.py). GRPO rollouts produce a
# different (batch, seq_len) shape almost every step -- unlike Phase 1
# SFT's fixed-length packed sequences -- so Dynamo recompiles per new shape
# and, once it exceeds its recompile cache (default limit 8), hard-fails
# instead of falling back to eager (fullgraph=True disables that fallback).
#
# First fix attempt -- raising torch._dynamo.config.recompile_limit right
# before trainer.train() -- did NOT work (job 37594245 hit the identical
# "currently set to 8" error). Root-caused by reading torch's own source
# (torch/_dynamo/eval_frame.py's `optimize()`): TORCHDYNAMO_DISABLE is
# checked at *decoration* time, i.e. when Unsloth's gemma.py patch wraps
# Gemma3RMSNorm.forward with torch.compile() during `import unsloth` --
# long before any config edit made later in this script's main() can have
# any effect. Must be set before that import, not after it.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

try:
    import unsloth  # noqa: F401  (import order matters; FastLanguageModel/PatchFastRL imported again, cheaply, where used)
except ImportError as e:
    raise ImportError(
        "Unsloth is required for this training path (pip install unsloth)."
    ) from e

import torch
from datasets import load_dataset, Dataset
from trl import GRPOTrainer, GRPOConfig

from core.schema import STREAM_SYSTEM_PROMPT
from core.parsing import parse_streams
from core.rewards import (
    reward_format, reward_semantic_fidelity, reward_physio_grounding,
    reward_hypergraph_bound, reward_tool_call, reward_empathy,
    reward_metacognitive_selfcorrection, reward_context_retention,
    reward_forecast_accuracy, reward_theory_of_mind,
)
from core.hypergraph.verification import InterimRuleBasedChecker, LearnedHypergraphChecker
from training.sft import GradientCollapseDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_content(completion) -> str:
    """TRL completions can be either a raw string or a conversational
    message-dict list -- this normalizes both to plain text, matching the
    handling pattern in Ahmed's own prior reward functions.
    """
    if isinstance(completion, list):
        return completion[0]["content"]
    return completion


# ---------------------------------------------------------------------------
# TRL-compatible reward function factories
#
# Each returns a function matching TRL's `(completions, **kwargs) -> list[float]`
# signature. Dataset columns named exactly `reference_patient_state`,
# `recipient_type`, `must_mention_facts`, and `true_future_values` (if present
# in the training dataset) are auto-passed by TRL as same-named kwargs,
# broadcast to match completions-per-prompt automatically.
# ---------------------------------------------------------------------------

_DEBUG_REWARD_LOGGING = os.environ.get("MDT_DEBUG_REWARD_LOGGING", "0") == "1"


def make_format_reward_func():
    def format_reward_func(completions, **kwargs) -> list[float]:
        extracted = [_extract_content(c) for c in completions]
        rewards = [reward_format(c) for c in extracted]
        if _DEBUG_REWARD_LOGGING:
            logger.warning("=== DEBUG format_reward_func: %d completions ===", len(completions))
            logger.warning("=== DEBUG raw completions[0] type=%s repr(first 500)=%r ===",
                            type(completions[0]).__name__, repr(completions[0])[:500])
            logger.warning("=== DEBUG extracted[0] type=%s repr(first 500)=%r ===",
                            type(extracted[0]).__name__, repr(extracted[0])[:500])
            logger.warning("=== DEBUG rewards=%s ===", rewards)
        return rewards
    return format_reward_func


def make_semantic_reward_func():
    def semantic_reward_func(completions, reference_patient_state=None, **kwargs) -> list[float]:
        refs = reference_patient_state or [""] * len(completions)
        return [reward_semantic_fidelity(parse_streams(_extract_content(c)), ref)
                for c, ref in zip(completions, refs)]
    return semantic_reward_func


def make_physio_reward_func():
    def physio_reward_func(completions, **kwargs) -> list[float]:
        return [reward_physio_grounding(parse_streams(_extract_content(c))) for c in completions]
    return physio_reward_func


def make_hypergraph_bound_reward_func(hypergraph_checker):
    """Closure over the checker (interim or learned) so the reward function
    itself matches TRL's plain (completions, **kwargs) signature."""
    def hypergraph_bound_reward_func(completions, **kwargs) -> list[float]:
        return [reward_hypergraph_bound(parse_streams(_extract_content(c)), hypergraph_checker)
                for c in completions]
    return hypergraph_bound_reward_func


def make_tool_call_reward_func():
    def tool_call_reward_func(completions, **kwargs) -> list[float]:
        return [reward_tool_call(_extract_content(c)) for c in completions]
    return tool_call_reward_func


def make_empathy_reward_func():
    def empathy_reward_func(completions, recipient_type=None, **kwargs) -> list[float]:
        types = recipient_type or ["clinician"] * len(completions)
        return [reward_empathy(parse_streams(_extract_content(c)), rt)
                for c, rt in zip(completions, types)]
    return empathy_reward_func


def make_metacognitive_reward_func():
    def metacognitive_reward_func(completions, **kwargs) -> list[float]:
        return [reward_metacognitive_selfcorrection(parse_streams(_extract_content(c))) for c in completions]
    return metacognitive_reward_func


def make_retention_reward_func():
    def retention_reward_func(completions, must_mention_facts=None, **kwargs) -> list[float]:
        facts_list = must_mention_facts or [[]] * len(completions)
        return [reward_context_retention(parse_streams(_extract_content(c)), facts)
                for c, facts in zip(completions, facts_list)]
    return retention_reward_func


def make_forecast_reward_func():
    def forecast_reward_func(completions, true_future_values=None, **kwargs) -> list[float]:
        values_list = true_future_values or [{}] * len(completions)
        return [reward_forecast_accuracy(parse_streams(_extract_content(c)), values)
                for c, values in zip(completions, values_list)]
    return forecast_reward_func


def make_tom_reward_func():
    # ADDED 2026-08-13 alongside core/rewards/theory_of_mind.py. Dataset
    # columns recipient_knows/recipient_does_not_know are populated by
    # core/cohort/grpo_prompts.py's _sample_recipient_knowledge_state --
    # same auto-pass-by-column-name mechanism TRL already uses for
    # recipient_type/must_mention_facts/true_future_values above. If a
    # prompt dataset predates this addition and lacks these columns, TRL
    # simply never passes the kwargs, so they default to None here and
    # reward_theory_of_mind returns its neutral 1.0 -- no crash, no
    # silent corruption of older datasets.
    def tom_reward_func(completions, recipient_knows=None, recipient_does_not_know=None,
                         **kwargs) -> list[float]:
        knows_list = recipient_knows or [None] * len(completions)
        does_not_know_list = recipient_does_not_know or [None] * len(completions)
        return [reward_theory_of_mind(parse_streams(_extract_content(c)), knows, does_not_know)
                for c, knows, does_not_know in zip(completions, knows_list, does_not_know_list)]
    return tom_reward_func


# ---------------------------------------------------------------------------
# Model loading (Unsloth path, matching the proven working setup)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer_unsloth(base_model_name: str, max_seq_length: int = 2048,
                                      lora_rank: int = 32, load_in_4bit: bool = True):
    """Loads via Unsloth's FastLanguageModel, matching the exact pattern
    already proven to work in Ahmed's prior repository. Falls back to
    raising a clear error if Unsloth isn't installed rather than silently
    trying a different (untested-in-this-combination) loading path.
    """
    try:
        from unsloth import FastLanguageModel, PatchFastRL
    except ImportError as e:
        raise ImportError(
            "Unsloth is required for this training path (pip install unsloth). "
            "If you need to train without Unsloth, use a plain transformers+peft "
            "loading path instead, but note the GRPO integration here follows the "
            "Unsloth-specific pattern validated in the prior repository, and has "
            "not been separately validated without it."
        ) from e

    # See training/sft.py's matching comment: Unsloth's ahead-of-time
    # compiled LoRA forward (unsloth_compiled_cache/Linear_peft_forward.py,
    # regenerated fresh on every cold model load) references peft's
    # `VARIANT_KWARG_KEYS` without importing it. Injecting into builtins
    # survives cache regeneration; patching the generated file does not.
    import builtins
    from peft.tuners.lora.layer import VARIANT_KWARG_KEYS as _VARIANT_KWARG_KEYS
    builtins.VARIANT_KWARG_KEYS = _VARIANT_KWARG_KEYS

    logger.info("Patching RL algorithms for GRPO/Unsloth compatibility ...")
    PatchFastRL("GRPO", FastLanguageModel)

    logger.info("Loading %s via Unsloth FastLanguageModel ...", base_model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto-detect
        load_in_4bit=load_in_4bit,
        fast_inference=False,  # Python 3.13 / environment compatibility, per prior working setup
    )

    # Both of Unsloth's own merge paths for handing off Phase 1's trained
    # adapter to Phase 2 (save_pretrained_merged(save_method="merged_16bit"),
    # and separately PEFT's own merge_and_unload()) were confirmed via
    # scripts/validate_phase1_checkpoint.py to silently corrupt the model --
    # the merged checkpoint never emits the trained <think>/<patient_state>/
    # <forecast>/<user_belief> format at all (0/4 compliant, degenerate
    # output on some prompts), while loading the SAME adapter directly (no
    # merge) is 4/4 compliant with genuinely good clinical content. Root
    # cause not fully pinned down in Unsloth's merge internals (per-layer
    # _merge_lora math checked out correctly in isolation for one attention
    # layer, so the bug is elsewhere in its broader save/shard pipeline) and
    # PEFT's merge_and_unload() + save_pretrained() left at least one
    # parameter un-materialized (reload hit `NotImplementedError: Cannot
    # copy out of meta tensor; no data!`, and the saved file was suspiciously
    # undersized versus a fully-materialized bf16 4.36B model).
    #
    # Given loading the adapter directly is proven correct, Phase 2 now
    # skips merging entirely and continues training Phase 1's SAME adapter
    # with GRPO instead of merging into a full model and attaching a fresh
    # second one. This is a legitimate, common RLHF pattern (same LoRA,
    # first SFT then RL objective) and sidesteps the whole class of merge-
    # correctness bugs above for good -- including for future backbones.
    # Confirmed via Unsloth's own loader.py: when `base_model_name` is an
    # adapter checkpoint (adapter_config.json present), it already returns
    # the adapter loaded with is_trainable=True and gradient-checkpointing
    # hooks patched (dispatch_model.patch_peft_model) -- so no further
    # get_peft_model() call is needed or wanted here; calling it anyway
    # would attach a REDUNDANT second LoRA on top of the first.
    is_adapter_checkpoint = (Path(base_model_name) / "adapter_config.json").exists()
    if is_adapter_checkpoint:
        logger.info(
            "%s is a LoRA adapter checkpoint -- continuing training on this SAME "
            "adapter (no merge, no fresh LoRA attached).", base_model_name
        )
        return model, tokenizer

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # See training/sft.py's matching comment: MedGemma's SigLIP
        # vision tower shares q_proj/k_proj/v_proj leaf names with the
        # language model, so the plain target_modules list above would
        # silently attach LoRA to it too. Unsloth's own documented
        # finetune_vision_layers=False mechanism for this exact situation
        # was tested directly (with and without the UNSLOTH_USE_NEW_MODEL=1
        # env var it needs in some code paths) and had no measurable effect,
        # so this excludes the vision tower via peft's own exclude_modules
        # regex instead, independent of Unsloth's regex-builder abstraction.
        exclude_modules=r".*vision_tower.*",
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Dataset formatting: builds the {"prompt": [...], + reward-relevant columns}
# structure GRPOTrainer + the reward functions above expect.
# ---------------------------------------------------------------------------

def format_dataset_for_grpo(dataset_path: str, bos_token: str = "") -> Dataset:
    """Loads the Phase-1-bridged prompt dataset (core/cohort/grpo_prompts.py's
    output) and reshapes it into TRL's expected {"prompt": [messages]} format,
    keeping reference_patient_state / recipient_type / must_mention_facts /
    true_future_values as extra columns TRL will auto-pass to reward funcs.
    """
    import pandas as pd
    if dataset_path.endswith(".parquet"):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_json(dataset_path, lines=True)

    required_cols = {"prompt", "reference_patient_state", "recipient_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"GRPO prompt dataset missing required columns: {missing}. "
            f"Run core/cohort/grpo_prompts.py against your Phase 1 vignette output first."
        )

    # Confirmed via smoke test (job 37774384): building this as a
    # conversational message list made every single format-dependent
    # reward exactly 0.0 (zero mean, zero variance) across the whole
    # batch -- TRL's GRPOTrainer detects the {"role": ..., "content": ...}
    # list via is_conversational() and auto-applies the tokenizer's chat
    # template (data_utils.py's maybe_apply_chat_template), wrapping the
    # prompt in Gemma's <start_of_turn>user/model turn structure. Phase 1
    # was never trained on that structure -- training/sft.py's
    # format_reasoning_example/format_synthetic_example both build a plain
    # raw-text template (STREAM_SYSTEM_PROMPT + "### Input" + "### Response"),
    # and the model was taught to continue "### Response\n" with "<think>",
    # not "<start_of_turn>model\n". Building the SAME raw string here (not
    # a message list) keeps GRPOTrainer's is_conversational() check False,
    # so it tokenizes this directly instead of chat-templating it -- see
    # trl/data_utils.py's is_conversational/maybe_apply_chat_template.
    #
    # bos_token prefix: root-caused via debug logging inside the reward
    # functions (job 37874404) -- real rollout completions were coherent
    # on-topic prose but NEVER contained any stream tags at all, while an
    # identical standalone diagnostic (diag_tmp/diagnose_grpo_generation.py)
    # reliably produced them. The difference: TRL's GRPOTrainer tokenizes
    # prompts with `add_special_tokens=False` (trl/trainer/grpo_trainer.py's
    # self.processing_class(..., padding_side="left", add_special_tokens=
    # False) call) -- meaning rollout generation NEVER gets a <bos> token
    # prepended, while training/sft.py's tokenization used the
    # tokenizer's default add_special_tokens=True for every single Phase 1
    # example. The model was trained exclusively on sequences starting with
    # <bos> and had never once seen an input without it -- explaining
    # coherent-but-structurally-blind output (the model still understands
    # the domain/language, but the specific fine-tuned stream-tag behavior,
    # learned only in <bos>-prefixed contexts, doesn't transfer to a shifted
    # embedding/position context). Can't change TRL's hardcoded
    # add_special_tokens=False, so the fix prepends the BOS token as a
    # literal string INTO the prompt text itself, which survives tokenization
    # regardless of add_special_tokens.
    def to_prompt_text(prompt_text):
        return f"{bos_token}{STREAM_SYSTEM_PROMPT}\n\n### Input\n{prompt_text}\n\n### Response\n"

    df = df.copy()
    df["prompt"] = df["prompt"].apply(to_prompt_text)
    if "must_mention_facts" not in df.columns:
        df["must_mention_facts"] = [[] for _ in range(len(df))]
    if "true_future_values" not in df.columns:
        df["true_future_values"] = [{} for _ in range(len(df))]
    # recipient_knows/recipient_does_not_know (R_tom, added 2026-08-13):
    # None (not []) when absent -- reward_theory_of_mind treats None
    # specifically as "no ground truth available" (neutral 1.0), whereas an
    # empty list is a real, meaningful claim ("this recipient knows/doesn't
    # know nothing"). A prompt dataset built before this addition should be
    # scored as "unknown", not "recipient knows nothing".
    if "recipient_knows" not in df.columns:
        df["recipient_knows"] = [None for _ in range(len(df))]
    if "recipient_does_not_know" not in df.columns:
        df["recipient_does_not_know"] = [None for _ in range(len(df))]

    return Dataset.from_pandas(df, preserve_index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="google/medgemma-4b-it")
    parser.add_argument("--prompt_dataset", required=True,
                        help="Output of core/cohort/grpo_prompts.py")
    parser.add_argument("--output_dir", default="./checkpoints/phase2_grpo")
    parser.add_argument("--hypergraph_mode", choices=["interim", "learned"], default="interim")
    parser.add_argument("--hypergraph_path", default="")
    # Confirmed via direct GPU diagnostics (diag_tmp/diagnose_grpo_generation.py):
    # 100% of real GRPO prompts (core/cohort/grpo_prompts.py's real MIMIC-IV
    # vignettes + STREAM_SYSTEM_PROMPT, which alone is 215 tokens) exceed
    # 512 tokens -- actual range 1008-1796, mean ~1414. The old
    # max_prompt_length=512 default silently truncated EVERY prompt before
    # it ever reached the "### Response\n" cue the model was trained to
    # continue from, so the model was never actually shown a complete
    # prompt during rollout generation -- it just kept hallucinating more
    # fake input data (fabricated vitals/labs, fabricated exam questions)
    # instead of answering, which is why every format-dependent reward was
    # exactly 0.0 with zero variance across every step of every smoke test.
    # max_seq_length (the model's own context window, separate from these
    # two per-call budgets) must be >= max_prompt_length + max_completion_
    # length or the model gets truncated internally too -- raised to match
    # Phase 1's own training max_seq_length=4096 for consistency.
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_rank", type=int, default=32)
    # Sequential OOM investigation on an L4 (22GB) narrowed these down from
    # their intended values (num_generations=4, max_prompt_length=2048,
    # per_device_train_batch_size=4) to increasingly tight ones -- reducing
    # batch size 4->1 barely helped (Gemma3's eager attention, `Unsloth:
    # Gemma3 does not support SDPA -- switching to fast eager`, scales with
    # sequence length far more than batch), and even after max_prompt_length
    # 2048->1792 (99.2% real-prompt coverage) plus completion_length
    # 1024->768 plus PYTORCH_CUDA_ALLOC_CONF=expandable_segments, it was
    # STILL short by ~200MB. Rather than keep trading training quality for
    # marginal memory, switched to a B200 (180GB, see slurm/phase2_grpo.
    # sbatch) and restored these to their proper values -- 4 generations is
    # the standard GRPO group size (2 is a bare minimum with weaker
    # advantage-estimation statistics), and 2048 covers 100% of real
    # prompts, not just 99.2%.
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=50)
    # KL_BETA (2026-08-12): previously never set, so GRPOConfig silently used
    # TRL's own default of beta=0.0 -- NO KL penalty against the reference
    # (pre-GRPO) policy at all, meaning nothing here constrained how far the
    # policy could drift from the SFT checkpoint except the reward signal
    # itself. Given this project's real history of repeated GRPO NaN
    # collapses (job 37933496 at step 12, and others -- see this file's
    # module-level comments and docs/development-log.md), an unconstrained
    # policy is a plausible contributing factor worth controlling for
    # directly rather than continuing to rely on an undocumented library
    # default. 0.01 matches the value used in a separate from-scratch
    # Nemotron-backbone experiment (../2026-04-24_nemotron-v1/), not
    # otherwise validated on THIS backbone/recipe -- treat as a reasonable
    # starting point to test, not a proven-correct value.
    parser.add_argument("--kl_beta", type=float, default=0.01,
                         help="KL penalty coefficient against the reference policy (GRPOConfig's beta). "
                              "0.0 disables the KL penalty entirely (TRL's own default, and this "
                              "project's previously-silent behavior).")
    # Added 2026-08-14: job 39279689 was CANCELLED (SIGNAL Terminated, not a
    # crash/OOM) at step 83/200 after 1d14.5h, well inside the 72h time
    # limit -- cause unclear (likely external scancel or preemption), but a
    # clean checkpoint-80 was saved (save_steps=5). Rather than lose ~17h of
    # GPU time, allow resuming from any HF Trainer checkpoint directory.
    parser.add_argument("--resume_from_checkpoint", default=None,
                         help="Path to a checkpoint-N directory (e.g. "
                              "./checkpoints/phase2_grpo_v2repro_base/checkpoint-80) "
                              "to resume trainer state (step count, optimizer, scheduler) from.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer_unsloth(
        args.base_model, max_seq_length=args.max_seq_length, lora_rank=args.lora_rank,
    )

    if args.hypergraph_mode == "interim":
        hypergraph_checker = InterimRuleBasedChecker()
    else:
        if not args.hypergraph_path:
            raise ValueError("--hypergraph_path required when --hypergraph_mode learned")
        hypergraph_checker = LearnedHypergraphChecker(args.hypergraph_path)

    # See format_dataset_for_grpo's matching comment: TRL's GRPOTrainer
    # tokenizes prompts with add_special_tokens=False, so <bos> must be
    # included as a literal string prefix in the prompt text itself, or
    # rollout generation runs on an input distribution (no leading <bos>)
    # the model never saw a single example of during Phase 1 SFT.
    bos_token = tokenizer.bos_token or ""
    if not bos_token:
        logger.warning("tokenizer.bos_token is empty/None -- GRPO rollouts will not get a "
                        "<bos>-prefixed prompt, which is the exact condition that caused "
                        "every format-dependent reward to be 0.0 in prior runs.")
    dataset = format_dataset_for_grpo(args.prompt_dataset, bos_token=bos_token)
    logger.info("Loaded %d GRPO training examples", len(dataset))
    if _DEBUG_REWARD_LOGGING:
        logger.warning("=== DEBUG bos_token=%r ===", bos_token)
        logger.warning("=== DEBUG dataset[0]['prompt'] type=%s repr(first 50)=%r repr(last 300)=%r ===",
                        type(dataset[0]["prompt"]).__name__, repr(dataset[0]["prompt"])[:50],
                        repr(dataset[0]["prompt"])[-300:])

    reward_funcs = [
        make_format_reward_func(),
        make_semantic_reward_func(),
        make_physio_reward_func(),
        make_hypergraph_bound_reward_func(hypergraph_checker),
        make_tool_call_reward_func(),
        make_empathy_reward_func(),
        make_metacognitive_reward_func(),
        make_retention_reward_func(),
        make_forecast_reward_func(),
        make_tom_reward_func(),
    ]

    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Job 37933496 (UNSLOTH_FORCE_FLOAT32 unset, caching enabled) produced a
    # genuine, IMPROVING reward signal for 11 clean steps (format=1.0 every
    # step, semantic/physio rewards trending up, grad_norm stable 0.24-0.43)
    # before collapsing to NaN at step 12 -- the same ~10-20-step collapse
    # signature Phase 1 hit without UNSLOTH_FORCE_FLOAT32 (see
    # slurm/phase1_sft.sbatch's bug log entry), where grad-norm-clipping/
    # warmup tuning ALONE was confirmed insufficient and the float32 patch
    # was the fix that actually worked. Re-enable the same env var here for
    # the same reason; when set, generation_kwargs disables KV caching to
    # avoid the matching dtype crash in Cache.update() (Q/K upcast to fp32
    # by the patch, cache buffer/V left in the original dtype). The earlier
    # "catastrophically slow" measurement for this combination (couldn't
    # finish step 0 in 4h) predates the prompt-truncation fix and the B200
    # migration -- both change the calculus enough to be worth re-measuring
    # empirically rather than continuing to avoid on stale evidence.
    #
    # Uses the module-level _UNSLOTH_FORCE_FLOAT32_REQUESTED constant, NOT a
    # fresh os.environ.get() here -- confirmed via job 38030835 that by this
    # point in main(), load_model_and_tokenizer_unsloth() has already called
    # FastLanguageModel.from_pretrained(), which unconditionally resets this
    # env var itself (see the module-level comment where the constant is
    # defined). Re-reading it here would silently see "0" even when "1" was
    # genuinely requested and the Gemma3 attention patches were genuinely
    # installed using that original value.
    _force_float32 = _UNSLOTH_FORCE_FLOAT32_REQUESTED
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        logging_steps=1,
        report_to="tensorboard",
        learning_rate=args.learning_rate,
        beta=args.kl_beta,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        # See training/sft.py's matching note: real, longer
        # sequences produced raw grad_norm spikes into the tens of thousands
        # during Phase 1 diagnostics on this same Unsloth+Gemma3+4bit+LoRA
        # stack. Tighter than HF's max_grad_norm=1.0 default as a defensive
        # measure for Phase 2's real (prompt, completion) sequences too.
        max_grad_norm=0.5,
        lr_scheduler_type="cosine",
        bf16=is_bf16_supported,
        fp16=not is_bf16_supported,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        # Caps disk usage when save_steps is set low (e.g. for frequent
        # checkpointing on a slow, walltime-constrained run) -- only the
        # latest checkpoints matter for resuming, so no need to keep every
        # one from a 200-step run.
        save_total_limit=3,
        use_vllm=False,  # Python 3.13 / environment compatibility, per prior working setup
        **({"generation_kwargs": {"use_cache": False}} if _force_float32 else {}),
    )
    if _force_float32:
        logger.warning(
            "UNSLOTH_FORCE_FLOAT32=1 -- KV caching disabled during generation to avoid the "
            "known Cache.update() dtype crash. Expect slower rollout generation than the "
            "cached-generation path; re-measuring whether this is still prohibitively slow "
            "now that prompts are correctly sized and this is running on a higher-memory GPU."
        )

    # Kept as a named variable (not inlined into callbacks=[...]) specifically
    # so its .collapsed flag is checkable after trainer.train() returns --
    # see the save-guard below.
    collapse_detector = GradientCollapseDetector(patience=3)
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        callbacks=[collapse_detector],
    )

    logger.info("Starting Phase 2 GRPO training (TRL GRPOTrainer, %d reward functions) ...",
                len(reward_funcs))
    logger.warning(
        "R_meta (metacognitive self-correction) is UNVALIDATED -- periodically sample "
        "checkpoints and compare against expert clinician annotation rather than trusting "
        "this reward component's scores at face value. See core/rewards/composite.py docstring."
    )

    # See this file's top-of-module comment: the actual fix for the Dynamo
    # recompile-limit crash (TORCHDYNAMO_DISABLE=1) has to be set before
    # `import unsloth`, not here -- a late config edit at this point in
    # main() was tried first and confirmed NOT to work (job 37594245).
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Confirmed necessary via job 37933496: GradientCollapseDetector stops
    # training by setting control.should_training_stop = True, which makes
    # trainer.train() RETURN NORMALLY (not raise) -- job 37933496 hit a real
    # NaN collapse at step 12/200, and this unguarded save block silently
    # wrote those NaN-corrupted weights to .../final, indistinguishable from
    # a genuine successful completion. Save to a clearly-marked path instead
    # so a collapsed run can never be mistaken for real Phase 2 output.
    if collapse_detector.collapsed:
        final_path = Path(args.output_dir) / "COLLAPSED_DO_NOT_USE"
        logger.error(
            "Training stopped due to a detected gradient collapse (%s). Saving to %s "
            "(NOT 'final') so this run can never be silently mistaken for a real, "
            "successful Phase 2 checkpoint. Do not deploy or continue training from this "
            "directory -- diagnose and fix the instability, then re-run.",
            collapse_detector.collapse_reason, final_path,
        )
    else:
        final_path = Path(args.output_dir) / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info("Phase 2 GRPO run finished. Model saved to %s", final_path)


if __name__ == "__main__":
    main()
