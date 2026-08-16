"""
scripts/validate_phase2_checkpoint.py

Functional validation of a Phase 2 GRPO checkpoint, in the same spirit as
scripts/validate_phase1_checkpoint.py but answering a different question.
Phase 1's validator asks "did SFT teach the structured format at all?".
This one asks "did the GRPO objective actually IMPROVE the model, without
breaking the format Phase 1 already taught it?" -- by running the SAME
held-out prompts through BOTH a Phase 1 checkpoint and the Phase 2
checkpoint under test, and comparing structural compliance plus the two
reward components that are self-contained enough to score without
MIMIC-specific reference data (R_format, R_physio -- see
training/rewards.py; R_semantic/R_hypergraph/R_retention/R_forecast all
need reference columns this script's free-form prompts don't have).

This is a lightweight, fast qualitative check -- for the full six-endpoint
quantitative evaluation against the real MIMIC-IV evaluation partition, use
scripts/run_evaluation.py instead. Use this script for a quick "is this
checkpoint sane" gut check (e.g. right after a training job produces a new
checkpoint, before committing to a multi-hour full evaluation run).
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
from training.rewards import reward_format, reward_physio_grounding

# Same prompt set as scripts/validate_phase1_checkpoint.py, deliberately --
# keeping them identical is what makes the Phase 1 vs Phase 2 comparison
# meaningful (same inputs, same scoring, only the checkpoint differs).
TEST_PROMPTS = [
    "A 45-year-old male presents with chest pain radiating to the left arm, "
    "diaphoresis, and nausea for the past 30 minutes. What is the most "
    "likely diagnosis and immediate management?",
    "Explain the mechanism by which ACE inhibitors can cause a dry cough.",
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


def _load_and_generate(model_path: str, max_new_tokens: int, load_in_4bit: bool) -> list[str]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=load_in_4bit,
        fast_inference=False,
    )
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_inference(model)

    generations = []
    for prompt in TEST_PROMPTS:
        text = f"{STREAM_SYSTEM_PROMPT}\n\n### Input\n{prompt}\n\n### Response\n"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        # use_cache=False for the same reason as validate_phase1_checkpoint.py
        # and training/grpo_trainer.py's GRPOConfig(generation_kwargs=
        # {"use_cache": False}): required alongside UNSLOTH_FORCE_FLOAT32=1
        # to avoid the index_copy_ dtype-mismatch crash (README bug log item
        # 19b). Harmless (just slower) if that env var isn't set.
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False,
        )
        generations.append(tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    # Free GPU memory before the caller loads the next checkpoint -- this
    # script loads two full models sequentially (Phase 1 baseline, then
    # Phase 2 under test), and without this a mid-size GPU can OOM on the
    # second load.
    del model
    import torch
    torch.cuda.empty_cache()
    return generations


def _score(generations: list[str]) -> dict:
    per_prompt = []
    for prompt, generated in zip(TEST_PROMPTS, generations):
        fmt = check_format(generated)
        fully_compliant = (
            all(fmt["tags_present"].values())
            and all(fmt["tags_nonempty"].values())
            and fmt["all_tags_present_and_ordered"]
            and fmt["forecast_parses"] in ("parsed_ok", "not_applicable")
        )
        parsed = parse_streams(generated)
        per_prompt.append({
            "prompt": prompt,
            "generated": generated,
            "format_check": fmt,
            "fully_compliant": fully_compliant,
            "reward_format": reward_format(generated),
            "reward_physio_grounding": reward_physio_grounding(parsed),
        })
    n_compliant = sum(r["fully_compliant"] for r in per_prompt)
    return {
        "per_prompt": per_prompt,
        "n_fully_compliant": n_compliant,
        "compliance_rate": n_compliant / len(TEST_PROMPTS),
        "mean_reward_format": sum(r["reward_format"] for r in per_prompt) / len(TEST_PROMPTS),
        "mean_reward_physio_grounding": sum(r["reward_physio_grounding"] for r in per_prompt) / len(TEST_PROMPTS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--phase2_checkpoint", required=True,
                        help="Phase 2 (GRPO) checkpoint to validate, e.g. "
                             "./checkpoints/phase2_grpo/checkpoint-130 or .../final")
    parser.add_argument("--phase1_baseline", default=str(project_root / "checkpoints" / "phase1_sft"),
                         help="Phase 1 checkpoint to compare against. Pass --no_baseline to skip "
                              "and only score the Phase 2 checkpoint.")
    parser.add_argument("--no_baseline", action="store_true",
                         help="Skip loading the Phase 1 baseline; only score --phase2_checkpoint.")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", default=str(project_root / "checkpoints" / "phase2_grpo_validation.json"))
    args = parser.parse_args()

    baseline_score = None
    if not args.no_baseline:
        print(f"=== Loading Phase 1 baseline: {args.phase1_baseline} ===")
        baseline_generations = _load_and_generate(args.phase1_baseline, args.max_new_tokens, args.load_in_4bit)
        baseline_score = _score(baseline_generations)

    print(f"=== Loading Phase 2 checkpoint under test: {args.phase2_checkpoint} ===")
    phase2_generations = _load_and_generate(args.phase2_checkpoint, args.max_new_tokens, args.load_in_4bit)
    phase2_score = _score(phase2_generations)

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'=' * 80}\n[{i}] PROMPT: {prompt}\n{'-' * 80}")
        if baseline_score is not None:
            b = baseline_score["per_prompt"][i]
            print(f"PHASE 1 (baseline) -- compliant={b['fully_compliant']} "
                  f"R_format={b['reward_format']:.2f} R_physio={b['reward_physio_grounding']:.2f}")
        p = phase2_score["per_prompt"][i]
        print(f"PHASE 2 (under test) -- compliant={p['fully_compliant']} "
              f"R_format={p['reward_format']:.2f} R_physio={p['reward_physio_grounding']:.2f}")
        print(f"PHASE 2 GENERATED:\n{p['generated']}")

    print(f"\n{'=' * 80}\nSUMMARY")
    if baseline_score is not None:
        print(f"Phase 1 baseline : {baseline_score['n_fully_compliant']}/{len(TEST_PROMPTS)} compliant "
              f"({baseline_score['compliance_rate']:.0%}), mean R_format={baseline_score['mean_reward_format']:.3f}, "
              f"mean R_physio={baseline_score['mean_reward_physio_grounding']:.3f}")
    print(f"Phase 2 under test: {phase2_score['n_fully_compliant']}/{len(TEST_PROMPTS)} compliant "
          f"({phase2_score['compliance_rate']:.0%}), mean R_format={phase2_score['mean_reward_format']:.3f}, "
          f"mean R_physio={phase2_score['mean_reward_physio_grounding']:.3f}")
    if baseline_score is not None:
        delta_format = phase2_score["mean_reward_format"] - baseline_score["mean_reward_format"]
        delta_physio = phase2_score["mean_reward_physio_grounding"] - baseline_score["mean_reward_physio_grounding"]
        print(f"Delta (Phase 2 - Phase 1): R_format={delta_format:+.3f}, R_physio={delta_physio:+.3f}")
        print("NOTE: a positive delta on these two components is a sanity signal, not proof of overall "
              "improvement -- R_semantic/R_hypergraph/R_retention/R_forecast (5 of GRPO's 9 reward terms) "
              "aren't scored here since they need MIMIC reference data these free-form prompts don't have. "
              "Use scripts/run_evaluation.py for the full picture.")

    output = {
        "phase1_baseline_checkpoint": None if args.no_baseline else args.phase1_baseline,
        "phase2_checkpoint": args.phase2_checkpoint,
        "phase1_baseline_score": baseline_score,
        "phase2_score": phase2_score,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {args.output_json}")
    print("NOTE: this checks STRUCTURAL format adherence + 2 of 9 reward components only, not full "
          "clinical content quality or the complete GRPO reward signal.")


if __name__ == "__main__":
    main()
