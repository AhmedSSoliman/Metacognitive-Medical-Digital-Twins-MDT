"""
core/rewards/empathy.py -- R_emp: structural recipient empathy.

Ported verbatim from the source repo's training/rewards.py, including the
Flesch-Kincaid readability helpers (count_syllables,
compute_flesch_kincaid_grade), which are only used by reward_empathy and so
travel with it rather than into a separate utility module.
"""

from __future__ import annotations

from core.parsing import ParsedStreams


# ---------------------------------------------------------------------------
# R_emp: structural recipient empathy (audience adaptation quality)
# ---------------------------------------------------------------------------

_LAY_TERM_SUBSTITUTIONS = {
    "hypotension": "low blood pressure", "tachycardia": "fast heart rate",
    "hyperlactatemia": "a buildup of lactate in the blood", "oliguria": "low urine output",
}

def count_syllables(word: str) -> int:
    """Simple syllable counter for English words."""
    word = word.lower().strip()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count = 1
    return count


def compute_flesch_kincaid_grade(text: str) -> float:
    """Computes a simplified Flesch-Kincaid Grade Level index for a given text."""
    import re
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    num_sentences = max(len(sentences), 1)
    
    words = re.findall(r'\b\w+\b', text)
    num_words = max(len(words), 1)
    
    total_syllables = sum(count_syllables(w) for w in words)
    grade = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59
    return max(0.0, float(grade))


def reward_empathy(parsed: ParsedStreams, recipient_type: str) -> float:
    """`recipient_type` in {"clinician", "patient", "family"}. For non-clinician
    recipients, rewards use of plain-language substitutes over raw jargon in
    <user_belief>; for clinicians, rewards concise, jargon-appropriate framing.
    Also incorporates Flesch-Kincaid grade level to calibrate readability constraints.
    """
    if parsed.user_belief is None:
        return 0.0
    text = parsed.user_belief.lower()
    
    # Compute readability grade level for the explanation block
    grade_level = compute_flesch_kincaid_grade(parsed.user_belief)

    if recipient_type == "clinician":
        # Reward conciseness, explicit acknowledgment, and professional density (grade level >= 10)
        length_score = 1.0 if len(text.split()) < 60 else 0.5
        mentions_clinician = 1.0 if any(w in text for w in ["clinician", "nurse", "physician", "provider"]) else 0.5
        readability_score = 1.0 if grade_level >= 10.0 else 0.5
        return 0.4 * length_score + 0.4 * mentions_clinician + 0.2 * readability_score
    else:
        # Reward plain lay term substitutions and accessible grade level (<= 8.0)
        jargon_count = sum(1 for term in _LAY_TERM_SUBSTITUTIONS if term in text)
        plain_count = sum(1 for plain in _LAY_TERM_SUBSTITUTIONS.values() if plain in text)
        term_score = 0.5
        if jargon_count + plain_count > 0:
            term_score = plain_count / (jargon_count + plain_count)
        readability_score = 1.0 if grade_level <= 8.0 else 0.0
        return 0.6 * term_score + 0.4 * readability_score
