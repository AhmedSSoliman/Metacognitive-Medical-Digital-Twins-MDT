"""
scripts/merge_phase1_checkpoint.py

Merges Phase 1 SFT's LoRA adapter into the base MedGemma weights and saves a
full 16-bit merged model, so Phase 2 GRPO can start from a model that has
actually learned the <think>/<patient_state>/<forecast>/<user_belief> stream
format taught in Phase 1 -- instead of running GRPO directly against the raw
base model, which has never seen that format at all (the reward functions in
training/rewards.py parse those tags and would score near-zero for basically
every rollout against an untrained base model). Fresh LoRA adapters for
Phase 2 are then attached on top of this merged checkpoint by
training/grpo_trainer.py itself, via --base_model pointing here.

Uses Unsloth's own documented merge API (model.save_pretrained_merged(...,
save_method="merged_16bit")) rather than a manual peft merge_and_unload(),
since the base model is loaded 4-bit quantized and a plain PEFT merge does
not handle merging LoRA deltas into quantized weights correctly.
"""

import argparse
import json
from pathlib import Path

import builtins
from peft.tuners.lora.layer import VARIANT_KWARG_KEYS as _VARIANT_KWARG_KEYS
builtins.VARIANT_KWARG_KEYS = _VARIANT_KWARG_KEYS

from unsloth import FastLanguageModel
from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True,
                         help="Phase 1 SFT output dir containing adapter_config.json "
                              "(e.g. ./checkpoints/phase1_sft or a specific checkpoint-N "
                              "subdirectory). The base model is read from the adapter's "
                              "own config -- no separate --base_model needed.")
    parser.add_argument("--output_dir", default="./checkpoints/phase1_sft_merged")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--use_unsloth_merge", action="store_true",
                         help="Use Unsloth's save_pretrained_merged instead of PEFT's "
                              "merge_and_unload. Confirmed broken for this checkpoint "
                              "(see comment above) -- kept only for comparison/debugging.")
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Unsloth's own save_pretrained_merged(save_method="merged_16bit") was
    # confirmed via direct GPU diagnostics to produce a checkpoint that
    # generates garbage / never emits the trained <think>/<patient_state>/
    # <forecast>/<user_belief> format at all, while the exact same adapter
    # loaded directly (not merged) generates correctly -- i.e. the merge
    # step itself corrupts something. Per-layer inspection of one attention
    # layer's actual lora_A/lora_B tensors and _merge_lora()'s internal math
    # checked out correctly in isolation, so the bug is somewhere else in
    # Unsloth's broader save/shard pipeline (not yet fully root-caused) --
    # rather than keep chasing it, switched to PEFT's own standard
    # merge_and_unload(), which has native bnb-4bit dequant+merge support
    # in the installed peft version (unlike when this project's original
    # comment about "plain PEFT merge does not handle quantized weights
    # correctly" was written, which is no longer accurate here).
    if args.use_unsloth_merge:
        model.save_pretrained_merged(args.output_dir, tokenizer, save_method="merged_16bit")
    else:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)

    # save_pretrained_merged()/tokenizer.save_pretrained() alone do not
    # produce a fully reloadable VLM checkpoint here: `args.adapter_path`
    # is checkpoints/phase1_sft, and training/sft_trainer.py deliberately
    # unwraps and saves ONLY the inner .tokenizer there (its own fix for a
    # different bug -- TRL misdetecting MedGemma as a VLM during text-only
    # training), never the full multimodal processor. So loading FROM that
    # local adapter path returns the same incomplete plain tokenizer here
    # too, and the merged output ends up missing preprocessor_config.json/
    # processor_config.json entirely -- confirmed by diffing against the
    # original cached base model's files, and by this exact fix failing on
    # its first attempt (tokenizer.save_pretrained() saved nothing new,
    # since `tokenizer` was never a ProcessorMixin to begin with here).
    # Fixed by loading and saving the FULL processor explicitly from the
    # original base model reference (read from the adapter's own config,
    # not hardcoded), independent of whatever Unsloth's adapter-path load
    # returned.
    with open(Path(args.adapter_path) / "adapter_config.json") as f:
        base_model_name = json.load(f)["base_model_name_or_path"]
    processor = AutoProcessor.from_pretrained(base_model_name)
    processor.save_pretrained(args.output_dir)

    # Real bug in the installed transformers==4.57.2 itself, not this
    # project's code: PreTrainedTokenizerBase._from_pretrained (tokenization_
    # utils_base.py) loads a LOCAL model's config.json via plain json.load()
    # (producing a dict), then does `_config.model_type` -- ATTRIBUTE access
    # on that dict, inside Mistral-variant special-casing logic -- which
    # raises `AttributeError: 'dict' object has no attribute 'model_type'`
    # for ANY local model directory (regardless of actual model_type; the
    # crash happens before the mistral-specific check is ever evaluated),
    # whenever config.json has a `transformers_version` field
    # <= the installed version. This never surfaced when loading
    # checkpoints/phase1_sft directly, since a LoRA-adapter-only directory
    # has no config.json of its own -- only once this script produces a
    # full merged model directory (which does) does the buggy code path
    # trigger. The guard is `if transformers_version and ...`, so removing
    # this one field avoids the buggy branch entirely without needing to
    # patch transformers itself or pin a different version.
    config_path = Path(args.output_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config.pop("transformers_version", None)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Merged Phase 1 checkpoint ({args.adapter_path}) -> {args.output_dir}")


if __name__ == "__main__":
    main()
