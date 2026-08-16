"""
core/rewards/theory_of_mind.py -- R_tom: theory-of-mind belief accuracy.

NEW component, added 2026-08-13 (not part of the original ten-reward set
ported from the source repo). Fills a real gap: core/rewards/empathy.py's
R_emp scores <user_belief> on writing STYLE only (word count, jargon
substitution, Flesch-Kincaid readability) -- it never checks whether
<user_belief> is an ACCURATE model of what the recipient actually already
knows. A response can be perfectly readable and completely wrong about the
recipient's knowledge state (e.g. re-explaining a diagnosis the family
already knows, or silently assuming a patient has seen raw lab numbers
they've never been shown) and R_emp would score it just as well as a
response that gets this right. R_tom is a distinct, additive check.

GROUND TRUTH: recipient_knows / recipient_does_not_know, sampled per
vignette by data/synthetic/vignette_generation.py's
_sample_recipient_knowledge_state (see that function's docstring for the
generation rules -- clinicians always know the diagnosis and raw numbers,
patients/family have a randomized 50/50 split on whether the diagnosis has
already been discussed with them, and nobody "knows" the prognosis, since
that's what the response is being asked to provide). This is SYNTHETIC
ground truth for the synthetic vignette set only -- there is no equivalent
field for the tier-one reasoning dataset (medical-o1-reasoning-SFT) or for
real MIMIC-IV-derived prompts, so this reward returns a neutral score
(see below) whenever the caller doesn't have real knowledge-state data to
check against, exactly like R_forecast's "not applicable" convention.

This is UNVALIDATED against real clinician judgment, same caveat as R_meta
(core/rewards/metacognitive.py) -- the synthetic knowledge-state split is a
reasonable simplifying assumption for training signal, not a clinically
validated model of real patient/family knowledge, which varies enormously
in practice and would need real chart-review/interview data to establish
properly.
"""

from __future__ import annotations

import numpy as np

from core.parsing import ParsedStreams
from core.rewards._encoder import get_sentence_encoder


def reward_theory_of_mind(parsed: ParsedStreams, recipient_knows: list[str] | None,
                           recipient_does_not_know: list[str] | None,
                           similarity_threshold: float = 0.55) -> float:
    """Scores <user_belief> on two axes, averaged:

    1. EFFICIENCY (avoid redundant over-explanation): for each fact in
       recipient_knows, <user_belief> should NOT read as if reintroducing
       it as new information. Checked by absence -- if the fact's phrasing
       (or a close embedding match) appears in <user_belief>, that's
       treated as a (mild) miss, since restating something the recipient
       already knows as if it were new is a real theory-of-mind error, not
       harmless redundancy.
    2. COVERAGE (correctly address the unknown): for each fact in
       recipient_does_not_know, <user_belief> SHOULD reflect awareness that
       this needs explaining (checked the same way core/rewards/retention.py
       checks must_mention_facts -- substring match, falling back to
       embedding similarity above similarity_threshold).

    Returns a neutral 1.0 if recipient_knows and recipient_does_not_know are
    both None/empty (caller has no real knowledge-state ground truth for
    this example -- e.g. tier-one reasoning data, which has no
    recipient concept at all) -- matches R_forecast's "not applicable"
    convention: absence of ground truth is not a penalty.
    """
    if not recipient_knows and not recipient_does_not_know:
        return 1.0
    if parsed.user_belief is None:
        return 0.0

    encoder = get_sentence_encoder()
    text = parsed.user_belief.lower()
    scores = []

    for fact in (recipient_knows or []):
        mentioned = fact.lower() in text
        if not mentioned:
            emb = encoder.encode([fact, parsed.user_belief], convert_to_numpy=True)
            cos_sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8))
            mentioned = cos_sim > similarity_threshold
        # Efficiency: NOT mentioning an already-known fact scores 1.0 (correct
        # theory of mind); mentioning it as if new scores 0.5, not 0.0 --
        # restating known information is a real but mild error, not as bad
        # as omitting something the recipient genuinely needs (below).
        scores.append(0.5 if mentioned else 1.0)

    for fact in (recipient_does_not_know or []):
        addressed = fact.lower() in text
        if not addressed:
            emb = encoder.encode([fact, parsed.user_belief], convert_to_numpy=True)
            cos_sim = float(np.dot(emb[0], emb[1]) / (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8))
            addressed = cos_sim > similarity_threshold
        scores.append(1.0 if addressed else 0.0)

    return float(np.mean(scores)) if scores else 1.0
