"""Soft-mandatory chain-of-thought alignment utilities.

Adds provenance-aware reasoning fields to mixed datasets:
- `think_synth`: synthetic or normalized reasoning text
- `think_source`: `gold` or `synthetic`
- `think_confidence`: [0,1] confidence/quality estimate
- `think_weight`: per-sample training weight used for SFT

This module implements a "soft mandatory" approach: every sample gets a
reasoning field, but synthetic reasoning is down-weighted relative to gold.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Dict, List, Optional, Tuple


_MEDICAL_CUES = {
    "sepsis", "infection", "lactate", "creatinine", "wbc", "hypotension",
    "tachycardia", "oxygen", "respiratory", "hemodynamic", "diagnosis",
    "differential", "treatment", "monitor", "risk", "clinical",
}


def _normalize_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _word_count(text: str) -> int:
    return len([w for w in re.split(r"\s+", _normalize_text(text)) if w])


def _quality_score(text: str, prompt: str, min_chars: int, min_words: int) -> float:
    """Estimate quality score for synthetic reasoning in [0, 1]."""
    txt = _normalize_text(text)
    if not txt:
        return 0.0

    words = [w.lower() for w in re.findall(r"[A-Za-z_]+", txt)]
    unique_ratio = (len(set(words)) / max(1, len(words))) if words else 0.0
    medical_hits = sum(1 for cue in _MEDICAL_CUES if cue in txt.lower())

    length_score = 1.0 if len(txt) >= min_chars and _word_count(txt) >= min_words else 0.4
    lexical_score = min(1.0, max(0.0, unique_ratio * 1.4))
    cue_score = min(1.0, medical_hits / 3.0)

    # Prompt coverage by token overlap.
    prompt_terms = {w.lower() for w in re.findall(r"[A-Za-z_]+", _normalize_text(prompt)) if len(w) > 3}
    text_terms = set(words)
    overlap = len(prompt_terms & text_terms) / max(1, len(prompt_terms))
    coverage_score = min(1.0, overlap * 2.0)

    return float(0.35 * length_score + 0.25 * cue_score + 0.20 * coverage_score + 0.20 * lexical_score)


def _template_synthetic_think(example: Dict) -> str:
    """Create a conservative synthetic reasoning scaffold from available fields."""
    prompt = _normalize_text(example.get("prompt", ""))
    patient_state = _normalize_text(example.get("patient_state", ""))
    user_belief = _normalize_text(example.get("user_belief", ""))

    parts = [
        "Clinical reasoning scaffold:",
        f"1) Problem framing: {prompt[:300]}" if prompt else "1) Problem framing: assess presenting complaint and context.",
        f"2) Physiologic interpretation: {patient_state[:220]}" if patient_state else "2) Physiologic interpretation: evaluate vitals/labs trends and severity.",
        "3) Differential prioritization: rank likely causes and immediate risks.",
        "4) Plan: outline next diagnostics and treatment priorities.",
    ]
    if user_belief:
        parts.append(f"5) Communication note: tailor explanation to user profile ({user_belief[:140]}).")

    return " ".join(parts)


def apply_soft_think_alignment(
    examples: List[Dict],
    config: Optional[object] = None,
) -> Tuple[List[Dict], Dict]:
    """Apply soft-mandatory CoT alignment with provenance and weighting."""
    if not examples:
        return [], {
            "total_examples": 0,
            "gold_examples": 0,
            "synthetic_examples": 0,
            "quality_pass_rate": 0.0,
            "avg_think_weight": 0.0,
        }

    config_dict = asdict(config) if (config is not None and is_dataclass(config)) else {}

    min_chars = int(config_dict.get("think_quality_min_chars", 80))
    min_words = int(config_dict.get("think_quality_min_words", 14))
    weight_gold = float(config_dict.get("think_weight_gold", 1.0))
    weight_synth_high = float(config_dict.get("think_weight_synth_high", 0.45))
    weight_synth_low = float(config_dict.get("think_weight_synth_low", 0.20))
    teacher_name = str(config_dict.get("think_teacher_model", "Qwen/Qwen3.5-4B"))

    aligned: List[Dict] = []
    gold_count = 0
    synthetic_count = 0
    quality_pass_count = 0
    weight_sum = 0.0

    for raw in examples:
        ex = dict(raw)

        source = str(ex.get("source", "unknown")).lower()
        think_existing = _normalize_text(ex.get("think") or ex.get("reasoning") or "")
        is_o1_like = ("o1" in source) or ("medical" in source and "o1" in source)

        if is_o1_like and think_existing:
            think_gold = think_existing
            think_synth = think_existing
            think_source = "gold"
            think_confidence = 1.0
            think_weight = weight_gold
            think_quality_pass = True
            quality_score = 1.0
            gold_count += 1
        else:
            think_gold = ""
            think_synth = think_existing if think_existing else _template_synthetic_think(ex)
            quality_score = _quality_score(
                think_synth,
                _normalize_text(ex.get("prompt", "")),
                min_chars=min_chars,
                min_words=min_words,
            )
            think_quality_pass = quality_score >= 0.55
            think_source = "synthetic"
            think_confidence = float(round(quality_score, 4))
            think_weight = weight_synth_high if think_quality_pass else weight_synth_low
            synthetic_count += 1
            quality_pass_count += int(think_quality_pass)

        final_think = think_gold if think_gold else think_synth

        # Canonical reasoning aliases used by existing training prompts.
        ex["think"] = final_think
        ex["reasoning"] = final_think

        # Provenance + quality metadata.
        ex["think_gold"] = think_gold
        ex["think_synth"] = think_synth
        ex["think_source"] = think_source
        ex["think_confidence"] = think_confidence
        ex["think_quality_pass"] = bool(think_quality_pass)
        ex["think_quality_score"] = float(round(quality_score, 4))
        ex["think_weight"] = float(round(think_weight, 4))
        ex["think_teacher_model"] = teacher_name if think_source == "synthetic" else "human_or_dataset_gold"

        aligned.append(ex)
        weight_sum += float(ex["think_weight"])

    summary = {
        "total_examples": len(aligned),
        "gold_examples": gold_count,
        "synthetic_examples": synthetic_count,
        "synthetic_quality_pass": quality_pass_count,
        "quality_pass_rate": round(quality_pass_count / max(1, synthetic_count), 4),
        "avg_think_weight": round(weight_sum / max(1, len(aligned)), 4),
        "teacher_model": teacher_name,
    }

    return aligned, summary
