# Agentic Metacognitive MDT (August 2026 restructure)

See `PROGRESS.md` for real, sourced status on all four phases (what's
done, current numbers, and how to run each phase). See
`Limitations_ResearchGaps_ProblemStatement_Significance.md` for the
project's research framing — the limitations of prior work this project
addresses, the specific research gaps, problem statement, and clinical/
methodological significance.

This is a structural reorganization of the working codebase at
`../Agentic-DT_V1-July/`, ported on 2026-08-12. **`../Agentic-DT_V1-July/`
remains the active repo** — it has the real git history, is where the
Phase 1 baseline investigation happened (see its `docs/hypothesis_log.md`
and `docs/development-log.md`), and is where live SLURM training jobs run.
This directory is a snapshot reorganized into a cleaner package layout;
it is not yet the repo of record unless/until that decision is made
explicitly.

## Why this structure

The load-bearing design decision, carried over from the source repo, is
the dependency boundary between `core/` and everything else:

**`core/` has zero dependency on torch, transformers, trl, unsloth, or
peft.** It imports only the standard library, numpy, scipy, and
lightweight packages like sentence-transformers. This mirrors a pattern
already used in the source repo (`models/stream_parsing.py` and
`training/sft_formatting.py` were deliberately split out of
torch-importing modules for exactly this reason — see their docstrings)
and is now applied consistently across the whole codebase.

This means:
- `core/`'s tests (schema, parsing, rewards, hypergraph, cohort, tools)
  run on a login node, in CI, or on a laptop — no GPU allocation needed.
- `training/` and parts of `evaluation/` require the full ML stack and
  only run inside a SLURM job on a GPU node (`import unsloth` alone
  requires a CUDA-visible device — see `training/sft.py`).
- Changing a shared definition (e.g. the `<patient_state>` schema in
  `core/schema.py`) propagates to SFT formatting, GRPO reward scoring,
  hypergraph verification, and evaluation without touching four separate
  copies of the same logic.

The dependency arrow is one-directional: `training/` and `evaluation/`
import from `core/`; `core/` never imports from them.

## Layout

- `core/` — schema, parsing, reward functions (one file per reward
  component), hypergraph construction/verification/validation, cohort
  extraction, tool dispatch. No heavy ML deps.
- `training/` — SFT (`sft.py`), GRPO (`grpo.py`), Phase 4 rollout-as-a-
  service (`rollout.py`, merging the source repo's
  `agents/rollout_service.py` and `scripts/run_phase4_scaled_training.py`
  — see that file's docstring for the merge rationale), backbone/model
  loading (`backbone.py`).
- `evaluation/` — one file per endpoint (topological fidelity, predictive/
  AUROC/AUPRC, retention/trajectory MAE, delta-embedding, communication),
  plus `report.py` (aggregation) and `run_evaluation.py` (CLI entry point).
- `sandbox/` — virtual EHR sandbox for integration testing.
- `tests/` — mirrors the package structure (`test_rewards/`,
  `test_hypergraph/`, `test_cohort/`, `test_tools/`, `test_training/`,
  `test_evaluation/`, `test_sandbox/`).
- `configs/` — YAML hyperparameters per phase and per backbone. Only
  `configs/backbones/medgemma_4b.yaml` is populated — MedGemma-4B is the
  only backbone validated against real GPU/data in this project as of
  this port. Other backbones (Nemotron, OctoMed) have been discussed but
  not committed to; do not add config files for them without an explicit
  decision to do so.
- `scripts/` — shell/SBATCH wrappers only (no Python drivers — those live
  under `training/`, e.g. `python -m training.rollout`).
- `data/` — `interim_constraints.json` and `dangerous_trajectories/` are
  placeholders; neither exists as real content in the source repo as of
  this port.

## Known placeholders (not yet implemented, ported honestly as stubs)

- `core/tools/fhir.py` — no FHIR-compatible summary generation exists in
  the source repo yet.
- `training/context_pruning.py` — no separable context-pruning
  implementation exists yet; the docstring notes where related logic
  currently lives inline.
- `evaluation/ablation.py` / `scripts/run_ablation.sh` — no dedicated
  pre-registered ablation runner exists yet.

## Status of the Phase 1 baseline (updated 2026-08-14)

**Resolved.** The source repo's `docs/hypothesis_log.md` documents an
investigation into why the originally-reported 97% format-compliance
result for `phase1_sft_v2` could not be reproduced from its own
unchanged recipe. Every fixed-cause hypothesis (dataset/recipe changes,
package version drift, base model weight drift, GPU hardware/driver
differences) was checked and rejected. The actual cause, confirmed
2026-08-14 via a 3-seed run-to-run variance check on the identical
recipe: **the recipe is highly seed-sensitive**, spreading from 46.7%
(seed=42) to 56.3% (seed=202) to 92.0% (seed=101) on otherwise identical
runs. There was no hidden bug — picking the best-performing seed is the
correct fix.

The working baseline is `checkpoints/phase1_sft_v2_reproduction`, whose
contents were swapped 2026-08-14 to the seed=101 weights (**92.0% format
compliance, n=300, confirmed**) — see
`checkpoints/phase1_sft_v2_reproduction/BASELINE_SWAP_NOTE.md`. The
original seed=42 weights are preserved at
`checkpoints/phase1_sft_v2_reproduction_seed42_archived/` for reference.
See `PROGRESS.md`'s Phase 1 section for the full investigation trail and
current numbers for all three seeds tested.
