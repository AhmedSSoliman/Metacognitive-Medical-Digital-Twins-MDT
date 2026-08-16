"""
training/sft.py

Phase 1 (Aim 1): supervised fine-tuning of the Multi-Stream architecture.

PORTING NOTE (2026-08-12): this file merges TWO source modules:
  - ../Agentic-DT_V1-July/training/sft_trainer.py  (the trainer itself)
  - ../Agentic-DT_V1-July/training/sft_formatting.py (the two pure-string
    example formatters + the APPEND_ANSWER_TO_THINK diagnostic toggle)

NOTE ON sft_formatting.py: the two example formatters were briefly merged
into this file during the port and then moved back out, because merging them
here silently required a GPU to import them (this module's `import unsloth`
runs on any import of it), destroying the very property they were split out
for. They live in training/sft_formatting.py; see that file's header for the
full rationale, reproduced in part below.

WHY sft_formatting.py IS NOT IN core/:
sft_formatting.py was split out of sft_trainer.py in the source repo for
exactly the same reason core/schema.py was split out of the model wrapper:
sft_trainer.py's first import is `import unsloth`, which raises on a machine
with no GPU, so pure string-formatting logic was untestable on a login node
or in CI. That rationale is about TESTABILITY, and it is fully satisfied by
keeping the formatters importable without touching the trainer -- it does not
by itself argue for core/.

Against putting them in core/: these two functions are not general stream
utilities. They encode the Phase 1 SFT TRAINING-EXAMPLE template
specifically -- which dataset fields map to which stream, that
general-reasoning examples get "not applicable" for <forecast>, and the
APPEND_ANSWER_TO_THINK diagnostic toggle that exists purely to A/B a
suspected SFT data bug. None of that is meaningful to parsing, rewards,
hypergraphs, cohorts, or tools; all of it is meaningful only to SFT data
prep. core/ is meant to be the shared, dependency-free foundation, not a
dumping ground for every torch-free file.

Resolution: they stay in training/sft_formatting.py, imported by this module.
tests/test_training/test_sft_formatting.py imports them from there and needs
neither torch nor a GPU, exactly as in the source repo.

(The target tree permits this: training/ is allowed non-heavy-dependency
files; the boundary rule is that core/ must not need torch, not that
training/ must.)

ORIGINAL training/sft_trainer.py MODULE DOCSTRING:

    Phase 1: Supervised fine-tuning of the Multi-Stream architecture.

    Trains on TWO data sources, kept structurally separate per the two-tier
    data policy (Manuscript 3):
      - Tier-one: FreedomIntelligence/medical-o1-reasoning-SFT (general medical
        reasoning style) + constructed synthetic vignettes covering the
        <think>/<patient_state>/<user_belief> format explicitly.
      - MIMIC-IV vignettes are NOT used here for text generation targets --
        they are reserved for Aim 3 hypergraph derivation and held-out
        evaluation only. This script will refuse to run if pointed at a
        MIMIC-derived dataset by mistake (see the guard in `load_sft_dataset`).

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

import argparse
import logging
import os
from pathlib import Path

# Pure string formatting -- see this file's header and
# training/sft_formatting.py's. Imported BEFORE unsloth deliberately; that
# module itself imports only core.schema, so it stays GPU-free.
from training.sft_formatting import (
    APPEND_ANSWER_TO_THINK,
    format_reasoning_example,
    format_synthetic_example,
)

# Unsloth must be imported before trl/transformers/peft/datasets -- this is
# Unsloth's own explicit runtime warning ("Unsloth should be imported before
# trl, transformers, peft to ensure all optimizations are applied"), which
# this file previously violated (unsloth was only imported later, inside
# main(), after trl/transformers were already imported at module top-level
# below). Confirmed via direct GPU diagnostics to matter for more than just
# performance: with the wrong order, real training runs against this exact
# model/LoRA/dataset setup silently ended up with 0 trainable parameters at
# `trainer.train()` time (Unsloth's own "Trainable parameters = 0 of
# 4,359,684,464" banner), while an otherwise-identical script that imports
# unsloth first showed the correct ~59.6M trainable parameters and real
# (if separately unstable -- see the max_grad_norm/warmup notes below)
# gradients. See the README's bug log for the full diagnostic trail.
#
# Tradeoff: unsloth itself requires a GPU to import at all (raises
# NotImplementedError("...You need a GPU.") otherwise), so this script can
# no longer be imported -- not even for `--help` -- on a non-GPU machine
# (e.g. a login node). Every real invocation of this script is a SLURM GPU
# job already, and no test file imports this module directly, so this
# tradeoff is accepted rather than working around it.
try:
    import unsloth  # noqa: F401  (import order matters; FastLanguageModel is imported again, cheaply, where used)
except ImportError as e:
    raise ImportError(
        "Unsloth is required (pip install unsloth) so this checkpoint is directly "
        "loadable by training/grpo_trainer.py's Phase 2 setup, which also uses Unsloth."
    ) from e

import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments, TrainerCallback
from trl import SFTTrainer, SFTConfig

# STREAM_SYSTEM_PROMPT now comes from core.schema (imported in Part 1 above),
# and format_reasoning_example/format_synthetic_example are defined in Part 1
# of this same file -- so the source repo's sys.path hack and its two
# intra-project imports are both gone.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientCollapseDetector(TrainerCallback):
    """Stops training early on two distinct real failure modes observed in
    this project's Unsloth/Gemma3/long-context setup, rather than letting a
    dead or corrupted run burn hours of GPU time undetected:

    1. grad_norm exactly 0.0 for `patience` consecutive logged steps -- a
       real Phase 1 run reported grad_norm=0.0 at every one of 40 logged
       points across 200 steps, with nothing else about the run looking
       abnormal from the console output or exit code; only caught by
       manually inspecting a checkpoint's trainer_state.json well after the
       fact. A single zero can be a legitimate coincidence (e.g. a clipped
       update that happens to round to exactly zero); several in a row is not.
    2. grad_norm or loss becomes NaN, even once -- observed after a raw
       (pre-clip) gradient spike as large as 3.26 million in one diagnostic
       run. Unlike the exactly-zero case, NaN is not given any patience:
       once the loss/gradient is NaN, the LoRA weights it gets applied to
       are permanently corrupted from that step forward, so every
       subsequent step is guaranteed-wasted compute, not just suspicious.

    See the README's gradient-instability bug log entry for the full
    diagnostic trail behind both of these.
    """
    def __init__(self, patience: int = 3):
        self.patience = patience
        self.consecutive_zero = 0
        # Confirmed necessary via job 37933496: this callback stops
        # trainer.train() by setting control.should_training_stop = True,
        # which makes .train() RETURN NORMALLY (not raise) -- so a caller
        # doing `trainer.train(); model.save_pretrained(final_path)`
        # unconditionally saves whatever the model's current weights are,
        # NaN-corrupted or not, into a path indistinguishable from a real
        # successful "final" checkpoint. This flag lets callers check WHY
        # training stopped before deciding whether to save/trust the
        # result. See training/grpo_trainer.py's matching check.
        self.collapsed = False
        self.collapse_reason = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control

        grad_norm = logs.get("grad_norm")
        loss = logs.get("loss")
        if (grad_norm is not None and grad_norm != grad_norm) or (loss is not None and loss != loss):
            # `x != x` is the classic NaN check -- avoids an extra math/numpy
            # import just for this.
            logger.error(
                "loss or grad_norm is NaN at step %s (loss=%s, grad_norm=%s) -- the model is "
                "permanently corrupted from this step forward. Stopping training immediately "
                "rather than continuing to waste GPU time. See the README's gradient-"
                "instability bug log entry.",
                state.global_step, loss, grad_norm,
            )
            control.should_training_stop = True
            self.collapsed = True
            self.collapse_reason = f"NaN loss/grad_norm at step {state.global_step}"
            return control

        if grad_norm is None:
            return control
        if grad_norm == 0.0:
            self.consecutive_zero += 1
            logger.warning(
                "grad_norm exactly 0.0 at step %s (%d/%d consecutive logged steps) -- "
                "gradients may have stopped flowing.",
                state.global_step, self.consecutive_zero, self.patience,
            )
            if self.consecutive_zero >= self.patience:
                logger.error(
                    "grad_norm has been exactly 0.0 for %d consecutive logged steps -- "
                    "stopping training early rather than burning further GPU time on a run "
                    "that is not learning. See the README's gradient-collapse bug log entry.",
                    self.patience,
                )
                control.should_training_stop = True
                self.collapsed = True
                self.collapse_reason = f"grad_norm=0.0 for {self.patience} consecutive steps, last at step {state.global_step}"
        else:
            self.consecutive_zero = 0
        return control


FORBIDDEN_TIER_TWO_MARKERS = ["mimic", "icu_stay", "hadm_id", "stay_id"]


def load_sft_dataset(tier_one_name: str, synthetic_vignette_path: str,
                      target_vignette_fraction: float = 0.08) -> Dataset:
    """Loads and combines tier-one SFT sources only. Raises if any input
    looks like it originates from MIMIC-IV / IC3 (tier-two), enforcing the
    two-tier data separation at load time, not just by convention.

    The synthetic (stream-format) vignettes are OVERSAMPLED (repeated) so
    they make up `target_vignette_fraction` of the final combined dataset.
    Without this, a small hand-written vignette set (e.g. 10-300 examples)
    gets diluted to well under 1% of a 20,000+-example reasoning dataset and
    the model has no real chance to learn the <think>/<patient_state>/
    <user_belief> structural format from it -- this was flagged as a real
    problem before this fix was added, not a hypothetical concern.
    """
    for suspect in [tier_one_name, synthetic_vignette_path]:
        lowered = suspect.lower()
        if any(marker in lowered for marker in FORBIDDEN_TIER_TWO_MARKERS):
            raise ValueError(
                f"Refusing to load '{suspect}' into the SFT (tier-one) pipeline: "
                f"it looks like it may be MIMIC-IV/IC3-derived (tier-two) data. "
                f"Tier-two data is reserved for hypergraph derivation and evaluation "
                f"only -- see Manuscript 3's two-tier data policy."
            )
    if not (0.0 < target_vignette_fraction < 1.0):
        raise ValueError(f"target_vignette_fraction must be in (0, 1), got {target_vignette_fraction}")

    logger.info("Loading tier-one reasoning dataset: %s", tier_one_name)
    # This dataset requires an explicit config name -- "en" for the English
    # split (there are also Chinese and "mix" configs). Confirmed against the
    # dataset's own TRL integration example; do not call load_dataset without
    # the config name or it will error or silently load the wrong split.
    reasoning_ds = load_dataset(tier_one_name, "en", split="train")

    logger.info("Loading synthetic stream-format vignettes: %s", synthetic_vignette_path)
    synthetic_ds = load_dataset("json", data_files=synthetic_vignette_path, split="train")

    n_reasoning = len(reasoning_ds)
    n_vignette = len(synthetic_ds)
    if n_vignette == 0:
        raise ValueError(f"No examples found in {synthetic_vignette_path} -- cannot proceed with zero vignettes.")

    # Solve for repeat count R such that:
    #   R * n_vignette / (n_reasoning + R * n_vignette) = target_vignette_fraction
    f = target_vignette_fraction
    repeat_count = max(1, round((f * n_reasoning) / (n_vignette * (1 - f))))

    logger.info(
        "Vignette oversampling: %d original vignettes, repeated %dx -> %d effective vignette "
        "examples, against %d reasoning examples (target fraction=%.1f%%, actual=%.1f%%)",
        n_vignette, repeat_count, n_vignette * repeat_count, n_reasoning,
        f * 100, 100 * (n_vignette * repeat_count) / (n_reasoning + n_vignette * repeat_count),
    )

    if repeat_count > 50:
        logger.warning(
            "Repeat count is %d -- your vignette set (%d examples) is very small relative to "
            "the reasoning dataset (%d examples). The model risks memorizing these %d exact "
            "examples verbatim rather than generalizing the stream format. Strongly consider "
            "growing the vignette set toward 150-300+ diverse examples instead of relying on "
            "oversampling alone to reach the target fraction.",
            repeat_count, n_vignette, n_reasoning, n_vignette,
        )

    # format_reasoning_example / format_synthetic_example now live in
    # training/sft_formatting.py (pure string logic, no unsloth/GPU
    # dependency -- see that module's docstring for why this split matters).
    reasoning_formatted = reasoning_ds.map(format_reasoning_example, remove_columns=reasoning_ds.column_names)
    synthetic_formatted = synthetic_ds.map(format_synthetic_example, remove_columns=synthetic_ds.column_names)

    oversampled_synthetic = concatenate_datasets([synthetic_formatted] * repeat_count)
    combined = concatenate_datasets([reasoning_formatted, oversampled_synthetic])
    combined = combined.shuffle(seed=42)
    logger.info("Combined SFT dataset: %d examples (%d reasoning + %d synthetic after %dx oversampling)",
                len(combined), len(reasoning_formatted), len(oversampled_synthetic), repeat_count)
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True,
                        help="e.g. google/medgemma-4b-it (validated in prior work), "
                             "AhmedSSoliman/octomed-7b-digital-twin-v1, openai/gpt-oss-20b")
    parser.add_argument("--tier_one_dataset", default="FreedomIntelligence/medical-o1-reasoning-SFT")
    parser.add_argument("--synthetic_vignettes", required=True,
                        help="Path to a JSONL file of constructed stream-format examples")
    parser.add_argument("--target_vignette_fraction", type=float, default=0.08,
                        help="Fraction of the final combined SFT dataset that should consist of "
                             "(oversampled) synthetic vignettes, so the stream-format signal isn't "
                             "diluted to near-zero by the much larger reasoning dataset. Default 8%%. "
                             "Used as Stage 2's fraction when --soft_alignment is enabled (default).")
    # --- Soft chain-of-thought alignment (docs/proposal.md, Phase 1) ---
    # The proposal specifies: "Rather than enforcing the [multi-]stream
    # format as a hard constraint from the first training step, we will
    # adopt a soft chain-of-thought alignment strategy: training examples
    # are weighted by provenance, so that fidelity to expert-authored
    # reasoning chains is preserved while the model is still permitted to
    # learn formatting flexibility from synthetic ICU trajectories... This
    # soft strategy is intended to avoid the brittleness that can result
    # from rigidly enforcing structural compliance before the underlying
    # reasoning capacity has been established." This was NOT implemented
    # in the original single-stage training loop below -- every example
    # (expert-authored reasoning chain or synthetic vignette) received
    # identical, full weight from step one, with no soft/gradual
    # introduction of the structural format at all.
    #
    # Implemented here as a two-stage curriculum rather than per-token loss
    # reweighting inside TRL's internal collation pipeline: Stage 1 trains
    # primarily on expert reasoning chains (a LOW vignette fraction --
    # format exposure present but not dominant, "permitted to learn
    # formatting flexibility" rather than forced), establishing baseline
    # reasoning capacity first; Stage 2 continues training the SAME LoRA
    # adapter (no merge, no fresh adapter -- the same proven continuation
    # pattern used for the Phase 1 -> Phase 2 handoff) with a HIGHER
    # vignette fraction to more strongly teach the structural format once
    # that reasoning capacity is established. A custom per-example loss
    # weight was considered but rejected as the primary mechanism: TRL's
    # SFTTrainer tokenizes and collates internally, and reliably threading
    # an extra per-example weight tensor through that pipeline without
    # live GPU testing to verify each TRL version's internal collator
    # behavior was judged too fragile to trust blind.
    parser.add_argument("--soft_alignment", action=argparse.BooleanOptionalAction, default=True,
                         help="Enable the two-stage soft chain-of-thought alignment curriculum "
                              "(default: on, per docs/proposal.md's Phase 1 spec). Pass "
                              "--no-soft_alignment for the original single-stage behavior.")
    parser.add_argument("--stage1_vignette_fraction", type=float, default=0.02,
                         help="Vignette fraction during Stage 1 (soft/low exposure) -- format "
                              "signal present but not dominant, so baseline reasoning capacity "
                              "from the expert-authored chains is established first.")
    parser.add_argument("--stage1_epochs", type=int, default=2,
                         help="Epochs of Stage 1 (low vignette fraction) -- only used when "
                              "--soft_alignment is on.")
    parser.add_argument("--stage2_epochs", type=int, default=1,
                         help="Epochs of Stage 2 (full --target_vignette_fraction, continuing the "
                              "same adapter from Stage 1) -- only used when --soft_alignment is on.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3,
                         help="Total epochs for the ORIGINAL single-stage behavior only "
                              "(--no-soft_alignment). Ignored when --soft_alignment is on -- use "
                              "--stage1_epochs / --stage2_epochs instead.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42,
                         help="Training seed, passed through to SFTConfig. Previously unset "
                              "(implicitly using HF's own default of 42) -- made explicit so "
                              "runs are reproducible and comparable.")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Uses Unsloth's FastLanguageModel (not the plain-PEFT MultiStreamModel
    # wrapper) so that Phase 1's saved checkpoint is directly loadable by
    # Phase 2's Unsloth-based GRPOTrainer setup -- these must use the same
    # underlying loading/LoRA mechanism, or Phase 2 may not be able to load
    # Phase 1's output at all. This was a real consistency gap caught while
    # wiring the two phases together (see README's bug/gap log).
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "Unsloth is required (pip install unsloth) so this checkpoint is directly "
            "loadable by training/grpo_trainer.py's Phase 2 setup, which also uses Unsloth."
        ) from e

    # Unsloth ahead-of-time-compiles each LoRA layer's forward pass into
    # unsloth_compiled_cache/Linear_peft_forward.py, generated fresh on every
    # cold model load. That generated code references peft's module-level
    # `VARIANT_KWARG_KEYS` constant (added in peft>=0.17 for LoRA variants
    # like aLoRA) without importing it -- an unsloth_zoo codegen bug against
    # current peft, causing a NameError on the model's first forward pass.
    # Patching the generated .py file directly does not survive: Unsloth
    # regenerates and overwrites it on the next cold load. Injecting the name
    # into builtins does survive, since Python's global lookup falls back to
    # builtins for any name not defined in the generated module's own
    # namespace, regardless of how many times that module gets regenerated.
    import builtins
    from peft.tuners.lora.layer import VARIANT_KWARG_KEYS as _VARIANT_KWARG_KEYS
    builtins.VARIANT_KWARG_KEYS = _VARIANT_KWARG_KEYS

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # MedGemma is a VLM (Gemma3 has a vision_config even for text-only
        # use). Its SigLIP vision tower ALSO has q_proj/k_proj/v_proj
        # submodules (just not o_proj/gate_proj/up_proj/down_proj), so the
        # plain target_modules list above silently attaches LoRA to 81
        # vision-tower attention modules that never see a gradient in this
        # text-only pipeline -- confirmed by inspecting trainable parameter
        # names after construction. Tried Unsloth's own
        # finetune_vision_layers=False/finetune_language_layers=True flags
        # (its documented mechanism for this exact situation) both with and
        # without the UNSLOTH_USE_NEW_MODEL=1 env var it requires in some
        # code paths -- neither had any measurable effect (identical
        # trainable-parameter count and names either way), so this excludes
        # the vision tower directly via peft's own exclude_modules regex
        # (a plain re.fullmatch against the full module path, independent of
        # Unsloth's regex-builder abstraction) instead of relying on a
        # library flag that does not appear to work in this Unsloth version.
        exclude_modules=r".*vision_tower.*",
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    if hasattr(tokenizer, "tokenizer"):
        # Gemma3 is natively multimodal (has a vision_config even for
        # text-only checkpoints), so Unsloth returns an AutoProcessor here
        # instead of a plain tokenizer. This pipeline is text-only (no
        # images -- see the two-tier data policy above), but if we hand the
        # processor straight to SFTTrainer, TRL detects `_is_vlm=True` from
        # `isinstance(processing_class, ProcessorMixin)` and expects
        # "messages"/"prompt"/"completion"/"images" columns instead of our
        # "text" column, which drops every column and crashes training.
        tokenizer = tokenizer.tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def build_sft_config(output_dir: str, num_train_epochs: int) -> SFTConfig:
        return SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            bf16=args.bf16,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=3,
            max_length=args.max_seq_len,
            dataset_text_field="text",
            # Explicit seed (was previously unset -- HF's default of 42 was
            # applied implicitly, but not recorded anywhere run-to-run). Set
            # explicitly here so results are reproducible and so a future
            # investigation into run-to-run variance (e.g. the v2 vs v3
            # checkpoint quality gap documented in docs/development-log.md)
            # can control for this directly rather than guessing whether two
            # runs used the same seed.
            seed=args.seed,
            report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
            # Do NOT also set gradient_checkpointing=True here. Unsloth's
            # get_peft_model(..., use_gradient_checkpointing="unsloth") above
            # already fully configures its own custom offloaded gradient
            # checkpointing, including the enable_input_require_grads() hook
            # required for gradients to flow from a frozen/quantized base model
            # into the trainable LoRA weights; also setting it here makes HF's
            # Trainer additionally call the standard
            # model.gradient_checkpointing_enable(), a redundant/conflicting
            # double-configuration. This alone turned out NOT to be the full
            # explanation for a real run that reported grad_norm=0.0 at every
            # logged step, though -- see the gradient-explosion note below,
            # confirmed as the actual root cause via direct GPU diagnostics.
            # NOT paged_adamw_8bit: switched to the standard (non-quantized)
            # optimizer as an ablation, given the raw pre-clip grad_norm
            # observed on real data is extremely unstable (fluctuating across
            # six orders of magnitude step to step, up to 1.4 million in one
            # diagnostic run) -- an 8-bit quantized optimizer's internal
            # moment/variance state is a plausible additional failure point
            # under that kind of input, on top of whatever upstream Gemma3/
            # Unsloth numerical issue produces the instability in the first place.
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            # Root cause, confirmed via direct diagnostics comparing toy vs. real
            # (medical-o1-reasoning-SFT, 600-950+ token) data on an otherwise
            # identical setup: real data reliably produces a raw (pre-clip)
            # grad_norm in the tens of thousands within the first couple of
            # steps (toy short-sequence data never does), most likely a Gemma3 /
            # Unsloth long-context numerical-stability issue (see
            # unsloth_zoo/temporary_patches/gemma.py's fp16 query/key path).
            # HF's default max_grad_norm=1.0 clips the applied update, but the
            # original warmup_ratio=0.03 (~60 steps of ramp over 2010 total)
            # meant that clipped-but-still-large update landed while the LR was
            # still ramping up from near-zero -- worsened, not caused, by a fast
            # ramp. Mitigated with a longer, gentler warmup and an explicit
            # (tighter than default) grad-norm cap, both standard, defensible
            # measures independent of ever fully root-causing the upstream
            # library's numerical behavior at this context length.
            max_grad_norm=0.5,
            warmup_ratio=0.1,
        )

    def run_stage(model, train_dataset, output_dir: str, num_train_epochs: int, stage_label: str):
        sft_config = build_sft_config(output_dir, num_train_epochs)
        # Kept as a named variable (not inlined into callbacks=[...]) specifically
        # so its .collapsed flag is checkable after trainer.train() returns --
        # see the save-guard below. Mirrors training/grpo_trainer.py's identical
        # guard, added there after job 37933496's NaN collapse at step 12/200
        # was silently saved to a path indistinguishable from a real successful
        # checkpoint; Phase 1 had the same unguarded trainer.save_model() call
        # and the same risk, just never yet triggered by an observed collapse.
        collapse_detector = GradientCollapseDetector(patience=3)
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[collapse_detector],
        )
        # Unsloth returns an EMPTY_LOGITS sentinel from the model's forward
        # pass instead of real logits (a deliberate memory-saving feature --
        # see unsloth_compiled_cache/unsloth_compiled_module_gemma3.py)
        # unless UNSLOTH_RETURN_LOGITS=1 is set. TRL's optional per-step
        # `entropy` and `mean_token_accuracy` metrics
        # (trl/trainer/sft_trainer.py's compute_loss, both gated by
        # `self.args.use_liger_kernel`) assume outputs.logits is always a
        # real tensor and crash on the sentinel. Forcing
        # UNSLOTH_RETURN_LOGITS=1 would materialize a (batch, seq_len,
        # vocab_size) float32 tensor -- for Gemma3's ~262k vocab that alone
        # risks OOM on the 22GB L4 GPU. Flipping this flag AFTER the
        # trainer is constructed disables only those two logging-only
        # metric blocks (dataset column prep, the flag's only other use in
        # this file, has already run) -- it does not touch the model, the
        # loss, or require the liger_kernel package.
        trainer.args.use_liger_kernel = True
        logger.info("Starting Phase 1 SFT training (%s) ...", stage_label)
        trainer.train()

        # See the collapse_detector construction comment above: without this
        # check, a collapsed/NaN-corrupted run would be saved to output_dir
        # indistinguishable from a genuine successful checkpoint.
        if collapse_detector.collapsed:
            save_dir = str(Path(output_dir) / "COLLAPSED_DO_NOT_USE")
            logger.error(
                "Training stopped due to a detected gradient collapse (%s). Saving to %s "
                "(not %s directly) so this run can never be silently mistaken for a real, "
                "successful Phase 1 checkpoint. Do not deploy or continue training from "
                "this directory -- diagnose and fix the instability, then re-run.",
                collapse_detector.collapse_reason, save_dir, output_dir,
            )
        else:
            save_dir = output_dir
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        logger.info("%s complete. Model saved to %s", stage_label, save_dir)
        return model

    if args.soft_alignment:
        logger.info(
            "Soft chain-of-thought alignment enabled (docs/proposal.md, Phase 1): Stage 1 "
            "(vignette_fraction=%.1f%%, %d epochs) establishes baseline reasoning capacity "
            "primarily from expert-authored chains before Stage 2 (vignette_fraction=%.1f%%, "
            "%d epochs) more strongly teaches the structural format, continuing the SAME "
            "adapter rather than starting fresh.",
            args.stage1_vignette_fraction * 100, args.stage1_epochs,
            args.target_vignette_fraction * 100, args.stage2_epochs,
        )
        stage1_dataset = load_sft_dataset(args.tier_one_dataset, args.synthetic_vignettes,
                                           target_vignette_fraction=args.stage1_vignette_fraction)
        stage1_dir = str(Path(args.output_dir) / "stage1")
        model = run_stage(model, stage1_dataset, stage1_dir, args.stage1_epochs,
                           "Phase 1 SFT Stage 1 (soft alignment)")

        stage2_dataset = load_sft_dataset(args.tier_one_dataset, args.synthetic_vignettes,
                                           target_vignette_fraction=args.target_vignette_fraction)
        model = run_stage(model, stage2_dataset, args.output_dir, args.stage2_epochs,
                           "Phase 1 SFT Stage 2 (format alignment)")
    else:
        train_dataset = load_sft_dataset(args.tier_one_dataset, args.synthetic_vignettes,
                                          target_vignette_fraction=args.target_vignette_fraction)
        run_stage(model, train_dataset, args.output_dir, args.num_train_epochs,
                  "Phase 1 SFT (single-stage)")


if __name__ == "__main__":
    main()
