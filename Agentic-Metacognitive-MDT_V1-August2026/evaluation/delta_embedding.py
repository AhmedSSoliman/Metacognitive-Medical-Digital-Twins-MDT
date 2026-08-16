"""
evaluation/delta_embedding.py

Delta-embedding concordance with expert annotation -- the audit endpoint for
R_meta (metacognitive self-correction).

STATUS: the delta-embedding COMPUTATION is real and fully implemented; the
CONCORDANCE-WITH-EXPERT-ANNOTATION comparison is a thin scaffold, because no
expert annotation set exists in the source repo yet.

RELATIONSHIP TO core/rewards/metacognitive.py: it is the same logic, and this
module deliberately does NOT duplicate it. `score_generations` below imports
and calls reward_metacognitive_selfcorrection directly, exactly as
evaluation/topological.py reuses reward_hypergraph_bound and
evaluation/communication.py reuses reward_empathy -- so the audit can never
drift from the reward it is auditing. See core/rewards/metacognitive.py's
docstring for the matching note.

WHY AN AUDIT ENDPOINT EXISTS AT ALL: R_meta is flagged throughout the source
repo as the one UNVALIDATED reward component. training/grpo.py emits a
runtime warning about it on every run ("R_meta ... is UNVALIDATED --
periodically sample checkpoints and compare against expert clinician
annotation rather than trusting this reward component's scores at face
value"), and core/rewards/composite.py weights it lower than the others
(w_meta=0.75) for the same reason. This module is where that periodic
comparison belongs.

WHAT IS MISSING: a set of generations whose <think> traces have been labelled
by clinicians for whether each pivot phrase reflects genuine reconsideration
versus superficial hedging. `concordance_with_expert_annotation` below takes
such labels and reports rank correlation, but there is no annotation file in
the project to feed it.
"""

from __future__ import annotations

import numpy as np

from core.parsing import parse_streams
from core.rewards.metacognitive import reward_metacognitive_selfcorrection


def score_generations(generations: list[str], window_tokens: int = 15) -> list[float]:
    """Delta-embedding score per generation, computed by the SAME function
    used as the R_meta training reward (imported, not reimplemented).
    Requires sentence-transformers at call time.
    """
    return [
        reward_metacognitive_selfcorrection(parse_streams(gen), window_tokens=window_tokens)
        for gen in generations
    ]


def concordance_with_expert_annotation(generations: list[str],
                                        expert_scores: list[float],
                                        window_tokens: int = 15) -> dict:
    """Spearman rank concordance between this system's delta-embedding scores
    and clinician-assigned scores for the same generations.

    Rank correlation (not Pearson, and not raw agreement) because the
    delta-embedding score is an unnormalized embedding-shift magnitude while
    an expert rating is typically an ordinal judgement -- the meaningful
    question is whether they ORDER the same traces the same way, not whether
    the numbers coincide.

    NOTE: untested against real data -- no expert annotation set exists in
    this project yet (see this module's docstring). Treat this as scaffolding
    to be validated when the first annotation batch arrives, not as a
    verified metric.
    """
    if len(generations) != len(expert_scores):
        raise ValueError(
            f"generations ({len(generations)}) and expert_scores ({len(expert_scores)}) "
            f"must be the same length -- each score must correspond to one generation."
        )
    if len(generations) < 2:
        return {"spearman_rho": float("nan"), "p_value": float("nan"), "n": len(generations)}

    from scipy.stats import spearmanr
    model_scores = score_generations(generations, window_tokens=window_tokens)
    rho, p = spearmanr(model_scores, expert_scores)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n": len(generations),
        "model_scores": model_scores,
        "model_score_mean": float(np.mean(model_scores)),
    }
