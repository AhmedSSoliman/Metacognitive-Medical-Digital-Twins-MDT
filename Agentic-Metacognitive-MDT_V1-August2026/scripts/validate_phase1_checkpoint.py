"""
scripts/validate_phase1_checkpoint.py

Functional validation of a Phase 1 SFT checkpoint: does the model actually
PRODUCE well-formed <think>/<patient_state>/<forecast>/<user_belief> output
at inference time? Phase 1's training-loss metrics (see README bug log /
checkpoints/phase1_sft/checkpoint-2010/trainer_state.json) confirm the run
converged cleanly, but a clean loss curve says nothing about whether the
model actually learned to EMIT the structured format Phase 2's reward
functions (training/rewards.py) and evaluation (evaluation/metrics.py)
depend on being able to parse. This script closes that gap with real
generation, not another loss number.

Runs generation over a small held-out prompt set spanning BOTH training
distributions -- general medical reasoning (matching medical-o1-reasoning-SFT)
and ICU-vignette style (two of these are real prompts held out from
data/synthetic/stream_format_vignettes.jsonl, not templated) -- then checks
structural format adherence via models.stream_parsing.parse_streams, the
exact same parser Phase 2's reward functions and evaluation use.

This is a STRUCTURAL check only: are all four tags present, non-empty, in
the required order, and does <forecast> parse as the constrained
'VAR_Nh: value [low-high]' sub-format. It does NOT judge clinical content
quality or correctness -- that needs expert review (see the R_meta caveat
already in training/rewards.py's docstring and the README). A model can
pass every check here while still being clinically wrong; this script only
answers "did SFT teach the format," not "is the model any good."
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import builtins
from peft.tuners.lora.layer import VARIANT_KWARG_KEYS as _VARIANT_KWARG_KEYS
builtins.VARIANT_KWARG_KEYS = _VARIANT_KWARG_KEYS

from unsloth import FastLanguageModel

from models.stream_parsing import STREAM_TAGS, STREAM_SYSTEM_PROMPT, parse_streams, parse_forecast_text

TEST_PROMPTS = [
    # General medical-reasoning style (matches medical-o1-reasoning-SFT domain,
    # ~92% of Phase 1's training mix) -- <patient_state>/<forecast> are
    # expected to legitimately say "not applicable" here, same as training.
    "A 45-year-old male presents with chest pain radiating to the left arm, "
    "diaphoresis, and nausea for the past 30 minutes. What is the most "
    "likely diagnosis and immediate management?",
    "Explain the mechanism by which ACE inhibitors can cause a dry cough.",
    # ICU-vignette style, real prompts held out from data/synthetic/
    # stream_format_vignettes.jsonl (NOT used in training, but same
    # distribution as the 36 vignettes that were) -- all four tags expected
    # to have real content, and <forecast> to use the numeric sub-format.
    "A 68-year-old male, 18 hours post-op from a bowel resection, has a "
    "heart rate of 112, MAP of 61, and a lactate that rose from 1.4 to 2.6 "
    "over the last 4 hours. Assess deterioration risk.",
    "A 45-year-old female with type 2 diabetes was admitted for DKA. "
    "Current glucose 210, pH 7.32, and she reports feeling 'much better' "
    "than admission. Summarize her current state for her family, who are "
    "anxious and have limited medical background.",
]


def check_format(generated_text: str) -> dict:
    parsed = parse_streams(generated_text)
    result = {"tags_present": {}, "tags_nonempty": {}, "forecast_parses": None}
    for tag in STREAM_TAGS:
        content = getattr(parsed, tag, None)
        result["tags_present"][tag] = content is not None
        result["tags_nonempty"][tag] = bool(content and content.strip())
    if result["tags_present"]["forecast"] and parsed.forecast:
        forecast_text = parsed.forecast.strip()
        if forecast_text.lower() == "not applicable":
            result["forecast_parses"] = "not_applicable"
        else:
            entries = parse_forecast_text(forecast_text)
            result["forecast_parses"] = "parsed_ok" if entries else "malformed"
    result["all_tags_present_and_ordered"] = _tags_in_order(generated_text)
    return result


def _tags_in_order(text: str) -> bool:
    positions = []
    for tag in STREAM_TAGS:
        idx = text.find(f"<{tag}>")
        if idx == -1:
            return False
        positions.append(idx)
    return positions == sorted(positions)


def main():
    parser = argparse.ArgumentParser()
    # Defaults are resolved relative to THIS FILE's location (not the
    # process's working directory) -- every sbatch job invokes this after
    # `cd`-ing into the project root, so a bare "./checkpoints/..." default
    # works there, but a direct `python path/to/validate_phase1_checkpoint.py`
    # from anywhere else (e.g. one level up, in MDT/) silently resolved
    # against the WRONG directory and produced "No config file found" --
    # confirmed by hitting exactly that running it from MDT/ instead of
    # MDT/Agentic-DT_V1-July/. Anchoring to __file__ makes the default work
    # regardless of caller's cwd.
    project_root = Path(__file__).resolve().parent.parent
    # Defaults point at the LoRA adapter checkpoint, not a merged model:
    # both Unsloth's save_pretrained_merged AND PEFT's merge_and_unload were
    # confirmed (via this exact script) to corrupt the merged output --
    # 0/4 format-compliant, degenerate output on some prompts -- while
    # loading the adapter directly is 4/4 compliant with genuinely good
    # clinical content. Phase 2 now continues training this same adapter
    # instead of merging (see training/grpo_trainer.py's matching comment),
    # so "the checkpoint to validate" is this one by default. Pass
    # --model_path explicitly to validate any other checkpoint.
    parser.add_argument("--model_path", default=str(project_root / "checkpoints" / "phase1_sft"))
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--output_json", default=str(project_root / "checkpoints" / "phase1_sft_validation.json"))
    # Adapter checkpoints are 4-bit base + adapter by construction (that's
    # how Phase 1 was trained), so default this to True -- the bf16-default
    # (False) was for isolating whether 4-bit re-quantization of an
    # already-MERGED model was the corruption source (it wasn't; ruled out
    # by job 37620696). Pass --load_in_4bit=False to override.
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        fast_inference=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)

    results = []
    n_fully_compliant = 0
    for i, prompt in enumerate(TEST_PROMPTS):
        text = f"{STREAM_SYSTEM_PROMPT}\n\n### Input\n{prompt}\n\n### Response\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        # use_cache=False deliberately matches training/grpo_trainer.py's
        # GRPOConfig(generation_kwargs={"use_cache": False}): with
        # UNSLOTH_FORCE_FLOAT32=1 set (required so this model's numerics
        # match what it was actually TRAINED under -- see slurm/phase1_sft.
        # sbatch's bug-log comment), use_cache=True crashes generation
        # entirely (index_copy_ dtype mismatch, see README bug log) -- this
        # is the exact bug that generation_kwargs fix addressed for Phase 2
        # rollouts, and the same fix applies here for the same reason.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=False,
        )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        fmt = check_format(generated)
        fully_compliant = (
            all(fmt["tags_present"].values())
            and all(fmt["tags_nonempty"].values())
            and fmt["all_tags_present_and_ordered"]
            and fmt["forecast_parses"] in ("parsed_ok", "not_applicable")
        )
        n_fully_compliant += int(fully_compliant)

        print(f"\n{'=' * 80}\n[{i}] PROMPT: {prompt}\n{'-' * 80}")
        print(f"GENERATED:\n{generated}\n{'-' * 80}")
        print(f"FORMAT CHECK: {json.dumps(fmt, indent=2)}")
        print(f"FULLY COMPLIANT: {fully_compliant}")

        results.append({
            "prompt": prompt,
            "generated": generated,
            "format_check": fmt,
            "fully_compliant": fully_compliant,
        })

    summary = {
        "n_prompts": len(TEST_PROMPTS),
        "n_fully_compliant": n_fully_compliant,
        "compliance_rate": n_fully_compliant / len(TEST_PROMPTS),
        "results": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}\nSUMMARY: {n_fully_compliant}/{len(TEST_PROMPTS)} prompts fully format-compliant "
          f"({summary['compliance_rate']:.0%})")
    print(f"Full results saved to {args.output_json}")
    print("NOTE: this checks STRUCTURAL format adherence only, not clinical content quality.")


if __name__ == "__main__":
    main()
