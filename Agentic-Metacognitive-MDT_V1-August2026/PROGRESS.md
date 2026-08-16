# Project Progress

Last updated: 2026-08-14

This file tracks real, verified status for the Agentic Metacognitive
Medical Digital Twin project — what's done, what the numbers actually are,
and what's next. All numbers below are pulled directly from evaluation
report JSON files (source: `../Agentic-DT_V1-July/evaluation/reports/`),
not estimates. See `../Agentic-DT_V1-July/docs/hypothesis_log.md` and
`docs/development-log.md` for the full investigation narrative behind
these numbers.

## TL;DR status

| Phase | Status | Current best result |
|---|---|---|
| Phase 1 (SFT) | **Resolved 2026-08-14**: the recipe is highly seed-sensitive, not regressed (see below). Baseline promoted to the best seed found so far. | **92.0% format compliance (n=300, seed=101, confirmed)** |
| Phase 2 (GRPO) | Job 39279689 (against the old seed=42 baseline) was CANCELLED at step 83/200 by an external signal, not a crash — clean checkpoint-80 saved. Resumed as job 39396398. | Old result (checkpoint-130, different baseline): 99.7% format compliance, 0.622 deterioration AUROC. New run's result pending. |
| Phase 3 (hypergraph) | Code-complete, real mining run completed; clinical review not yet started (0/4187, resumable review tool ready as of 2026-08-14) | 4187 candidate hyperedges, PENDING_CLINICAL_REVIEW |
| Phase 4 (scaled rollout) | Code-complete, never run against real GPU/data | Not yet run |

## Phases & how to run them

All commands below are `sbatch` SLURM jobs, run from this repo's root
(`Agentic-Metacognitive-MDT_V1-August2026/`). Each script's own comments
carry the full rationale for its current settings — this table is just the
practical "what does each phase do and how do I launch it" reference.

| Phase | What it does | How to run | Hyperparameters live in |
|---|---|---|---|
| **1 — SFT** | Supervised fine-tunes the base model (MedGemma-4B) on the tier-one reasoning dataset + hand-written ICU vignettes, teaching the four-stream `<think>/<patient_state>/<forecast>/<user_belief>` output format. | `sbatch scripts/run_sft.sh` | `configs/sft.yaml` |
| **2 — GRPO** | Multi-objective RL alignment on top of the Phase 1 checkpoint: builds GRPO prompts from the MIMIC-IV cohort, then runs group-relative policy optimization against 10 reward components (format, semantic fidelity, physio grounding, hypergraph boundary safety, tool use, empathy, metacognition, retention, forecast accuracy, diagnostic accuracy). | `sbatch scripts/run_grpo.sh` | `configs/grpo.yaml` |
| **3 — Hypergraph** | Mines statistically-significant co-occurrence hyperedges (e.g. `{tachycardia, hypotension, hyperlactatemia}`) from real MIMIC-IV vitals/labs, for use as the `R_bound` safety-boundary check in Phase 2/4. CPU-only, no GPU needed. Output is always versioned per job, never overwrites the reviewed file automatically. | `sbatch scripts/build_hypergraph.sh` | `configs/hypergraph.yaml` |
| **3b — Clinical review** | Manual, human-judgment step: a clinician approves/rejects each candidate hyperedge from Phase 3 via a CLI prompt. Cannot be automated. | `python scripts/clinical_review_ui.py <path-to-hyperedges.json>` (interactive, run directly, not via sbatch) | n/a |
| **4 — Scaled rollout** | Parallel rollout-worker pool + policy-gradient driver, for throughput/latency experiments at scale. Not yet run against real GPU/data. | `python -m training.rollout --phase2_checkpoint <path> --hypergraph_path <path> --prompt_dataset <path>` (no dedicated sbatch wrapper yet — required args per `training/rollout.py`'s `main()`) | n/a yet — no dedicated config file exists |
| **Evaluation** | Runs the full six-endpoint evaluation (format compliance, deterioration detection, topological fidelity, structural empathy, trajectory MAE, deployment feasibility) against any checkpoint, n=300 by default. | `sbatch scripts/run_evaluation.sh` (override checkpoint via `sbatch --export=CHECKPOINT=./checkpoints/other,OUTPUT_REPORT=./evaluation/reports/other.json scripts/run_evaluation.sh`) | `configs/evaluation.yaml` |
| **Chatbot UI** | Interactive Streamlit chat interface — parses a live generation into a normal chat response (see "Expected outputs" section 5 below). Not a training/eval job, run directly. | `streamlit run scripts/streamlit_app.py` | n/a |

**Recommended order for a from-scratch run:** Phase 1 → Evaluation (sanity
check the checkpoint) → Phase 3 (independent of Phase 1/2, can run any
time) → Phase 3b (clinical review, blocks `--hypergraph_mode learned`) →
Phase 2 (points `--base_model` at the Phase 1 checkpoint) → Evaluation
again (before/after comparison) → Phase 4.

## Phase 1: Supervised Fine-Tuning

### The open question: why doesn't the original 97% reproduce?

An earlier checkpoint (`phase1_sft_v2`) was evaluated at **97.0%** format
compliance (n=300, `topological_fidelity=0.973`). Every checkpoint trained
since — including retraining v2's own exact, unchanged recipe today — has
landed well below that:

| Checkpoint | format_compliance | deterioration_auroc | topological_fidelity | structural_empathy | n |
|---|---|---|---|---|---|
| `phase1_sft_v2` (original) | **97.0%** | 0.498 | 0.973 | 0.701 | 300 |
| `phase1_sft_v3` | 70.7% | 0.525 | 0.853 | 0.626 | 300 |
| `phase1_sft_v4` (seed=1337) | 20.7% | 0.531 | 0.833 | 0.589 | 300 |
| `phase1_sft_diag_noanswer` (1 epoch, confounded) | 9.7% | 0.515 | 0.223 | 0.172 | 300 |
| `phase1_sft_v2_reproduction` (v2's exact recipe, seed=42) | 46.7% | 0.576 | 0.927 | 0.636 | 300 (confirmed) |
| `phase1_sft_variance_seed101` (identical recipe, seed=101) | **92.0%** | — | — | — | **300 (confirmed) — NEW BASELINE** |
| `phase1_sft_variance_seed202` (identical recipe, seed=202) | 56.3% | — | — | — | 300 (confirmed) |

(For reference, the earlier n=30 smoke eval on the seed=42 checkpoint had
shown 66.7% / AUROC 0.716 — the full n=300 run landed notably lower on
both, which is itself informative: small-sample estimates on this
checkpoint were not reliable.)

**RESOLVED (2026-08-14): the v2-vs-v3+ gap was run-to-run seed variance,
not a regression.** The variance check settled it: identical recipe,
identical dataset, only `--seed` changed, and format compliance spread
from 46.7% (seed=42) to 56.3% (seed=202) to 92.0% (seed=101) — a 45-point
range from seed alone. This means the original 97% and the 46.7% baseline
were both just draws from a noisy distribution; there was no hidden bug
to find, and every earlier "REJECTED" hypothesis below was rejected
correctly (they were the wrong place to look — the recipe genuinely is
this seed-sensitive).

**Investigation summary** (full detail in
`../Agentic-DT_V1-July/docs/hypothesis_log.md`):
- Dataset expansion (36→1000 vignettes) — tested, not the sole cause (v2-reproduction used the original 36 and still regressed)
- The `SFT_APPEND_ANSWER_TO_THINK` behavior — tested at 1 epoch, confounded by under-training, inconclusive
- Training seed variance — **CONFIRMED as the actual explanation** (2026-08-14); the earlier "REJECTED" verdict below was based on a single seed=1337 comparison point (v4, 20.7%) that happened to also land low, masking the real spread until 2+ more seeds were tested
- Local package version drift — REJECTED (packages installed 2026-07-14, five days before v2's original 97% run, untouched since)
- Base model weight drift — REJECTED (single cached HF snapshot, commit `18ece6af...`, identical across every run)
- Evaluation/parsing code changes — checked, logic looks sound today; weak/unconfirmed lead (no real pre-2026-07-27 version exists to diff against)

**Decision (2026-08-14):** promote the seed=101 checkpoint (92.0%) to the
canonical `phase1_sft_v2_reproduction` baseline — its directory contents
were swapped in place (see
`checkpoints/phase1_sft_v2_reproduction/BASELINE_SWAP_NOTE.md`) so every
existing script/config pointing at that path now uses the new weights
with no reference changes needed. The original seed=42 weights are
preserved, unmodified, at
`checkpoints/phase1_sft_v2_reproduction_seed42_archived/`.

### What's next for Phase 1
- [x] Get the n=300 confirmation number for `phase1_sft_v2_reproduction` — **46.7%, confirmed 2026-08-12** (now superseded, see below)
- [x] GPU hardware/driver comparison, checked 2026-08-13: found the real
      original v2 training job (37504804, 14h runtime matching a full
      3-epoch run, node `c0608a-s4`) and confirmed its Unsloth startup
      banner (NVIDIA L4, Torch 2.13.0+cu130, CUDA 8.9, CUDA Toolkit 13.0)
      is IDENTICAL to every recent run. This lead was a dead end, but is
      now moot — the variance check below found the real answer.
- [x] **Run-to-run variance check — COMPLETE, RESOLVED 2026-08-14.** 3 runs
      of the identical recipe (seed=42/101/202) spread 46.7%-56.3%-92.0%.
      Seed=101 (92.0%) promoted to the new baseline (see Decision above).
      Seed=202's training job (39288831) hit an unrelated infrastructure
      Bus error on its first attempt (bad node, `c0611a-s13`) and was
      cleanly resubmitted (job 39289581) with no other changes.
- [ ] Optional future work: a 4th+ seed run, or averaging/ensembling
      across seeds, would further characterize the variance distribution,
      but is not required — the investigation's original question (is
      46.7% real or a regression?) is answered.

## Phase 2: GRPO Alignment

A full 200-step GRPO run completed against the **original, now-superseded**
Phase 1 checkpoint (before the v2/v3/v4 investigation began). Real,
n=300 evaluation numbers, with a genuine before/after-GRPO comparison:

| Checkpoint | format_compliance | topological_fidelity | structural_empathy | deterioration_auroc | deterioration_auprc |
|---|---|---|---|---|---|
| Phase 1 SFT (pre-GRPO baseline for this run) | 100.0% | 100.0% | 0.654 | 0.481 | 0.359 |
| Phase 2 checkpoint-130/200 | 99.7% | 99.7% | 0.793 | **0.622** | 0.485 |
| Phase 2 final-200/200 | 99.7% | 99.3% | 0.794 | 0.576 | 0.399 |

Notable: checkpoint-130 scored a *higher* deterioration AUROC than the
final 200-step checkpoint (0.622 vs 0.576), though confidence intervals
overlap substantially — reaching 200 steps required four independently-
restarted training segments (interrupted by NaN collapses and one genuine
SLURM node failure), which is a real methodological difference from an
uninterrupted run and a plausible contributor to the non-monotonic result.

**This run is now stale** — it was trained against a Phase 1 checkpoint
that predates the whole v2/v3/v4/v2-reproduction investigation.
`scripts/run_grpo.sh` has been updated to point at
`phase1_sft_v2_reproduction` instead, writing to a new output directory
(`phase2_grpo_v2repro_base`) so the existing, already-evaluated results
above are never overwritten.

**Job 39279689 (against the seed=42 baseline) was submitted 2026-08-12,
ran 1d14.5h, and was CANCELLED at step 83/200** — `sacct` showed
`SIGNAL Terminated`, not an OOM/crash, and there were no CUDA or memory
errors in the log; most likely an external `scancel` or a queue
preemption, well inside the 72h time limit. A clean `checkpoint-80`
(save_steps=5) was saved. Rather than lose ~17h of GPU time, added
`--resume_from_checkpoint` support to `training/grpo.py` and resubmitted
as **job 39396398**, resuming from checkpoint-80. Because checkpoint-80's
LoRA adapter was trained from the seed=42 base weights (not the
newly-promoted seed=101 baseline), the resume run's `--base_model` is
pinned to `phase1_sft_v2_reproduction_seed42_archived` specifically — see
`scripts/run_grpo.sh`'s `RESUME_CHECKPOINT` handling. A future *fresh*
GRPO run (no resume) should be started against the new seed=101 baseline
to get a GRPO result paired with the better Phase 1 checkpoint.

### What's next for Phase 2
- [x] Confirm the Phase 1 v2-reproduction n=300 eval looks reasonable — 46.7% format compliance, real number as of 2026-08-12 (superseded by seed=101's 92.0%, see Phase 1 section)
- [x] Submit `scripts/run_grpo.sh` against the (then-current) baseline — job 39279689, cancelled at step 83/200, resumed as job 39396398
- [ ] Once job 39396398 completes: re-run the same n=300 evaluation for a real before/after comparison against the numbers above
- [ ] Consider a fresh (non-resumed) GRPO run against the new seed=101 Phase 1 baseline, since job 39396398's reference policy is still the older seed=42 weights

## Phase 3: Hypergraph Construction

Code-complete and run successfully against real MIMIC-IV data:
**4187 candidate hyperedges** mined, status `PENDING_CLINICAL_REVIEW`.
Two real bugs were found and fixed getting here (see
`../Agentic-DT_V1-July/docs/development-log.md`, 2026-07-28 update):
vitals/labs weren't being combined before mining (meaning the flagship
`{tachycardia, hypotension, hyperlactatemia}` hyperedge could never
actually be mined), and a versioned-output-path fix to prevent a re-run
from silently clobbering a clinically-reviewed file.

### What's next for Phase 3
- [ ] **Manual clinical review** of the 4187 candidate hyperedges via
      `scripts/clinical_review_ui.py` — this is a human-judgment step
      that cannot be automated. **0/4187 reviewed as of 2026-08-14.**
      The source file (`hypergraph/derived_hypergraph_full_38777889.json`,
      copied 2026-08-14 from the source repo — this new repo's
      restructure never carried Phase 3's output over) is in place and
      ready to review:
      ```
      cd /blue/prismap-ai-core/Ahmed/DigitalTwins/MDT/Agentic-Metacognitive-MDT_V1-August2026
      python scripts/clinical_review_ui.py hypergraph/derived_hypergraph_full_38777889.json --reviewer "Ahmed"
      ```
      `scripts/clinical_review_ui.py` was extended 2026-08-14 with resume
      support (previously it had none — a Ctrl-C partway through would
      have lost all progress from that session): progress autosaves back
      into the same input file every 25 edges (`--autosave_every` to
      change), typing `q` at any prompt saves and exits cleanly, and
      re-running the same command skips every edge already marked
      `approved` and continues from where it left off. Verified via a
      scripted smoke test (autosave, resume-skip, and final promotion to
      `derived_hypergraph.json` all confirmed working) before being
      handed off for the real review.
- [ ] Promote the reviewed file to `hypergraph/derived_hypergraph.json`
      for use in Phase 2 (`--hypergraph_mode learned`) and Phase 4 — the
      script does this automatically once all 4187 edges are reviewed.

## Phase 4: Scaled Rollout

Code-complete (`training/rollout.py`, merging the source repo's
`agents/rollout_service.py` worker pool and
`scripts/run_phase4_scaled_training.py` driver) but **never run against
real GPU/data**. Two real bugs were found and fixed in the code before
this port (dead `get_recent_labs` tool, wrong prompt-dataset path in the
original sbatch script) but the fix has not yet been exercised end-to-end.

### What's next for Phase 4
- [ ] First real run, once Phase 2 has a checkpoint from the current baseline to roll out

## Recent additions (this repo only, not yet in the source repo)

- **F2-optimal threshold sweep** (`evaluation/predictive.py::find_f2_optimal_threshold`,
  wired into `run_full_evaluation`'s `EvaluationResult` as `f2_optimal_threshold`
  / `f2_at_optimal_threshold`). Reports the best achievable F2 for
  deterioration detection alongside the existing fixed-0.5-threshold
  numbers, never replacing them (post-hoc threshold selection on the eval
  set is optimistic and shouldn't be silently substituted for a
  pre-registered operating point). Inspired by a similar sweep in
  `../2026-06-15_nemotron-experiments/` (a separate, from-scratch
  Nemotron-backbone experiment, not otherwise related to this pipeline).
  3 new tests, 112/112 total passing.
- **Explicit `--kl_beta` for Phase 2 GRPO** (`training/grpo.py`,
  `scripts/run_grpo.sh`, `configs/grpo.yaml`). Previously `GRPOConfig`'s
  `beta` (KL penalty against the reference/pre-GRPO policy) was never set,
  silently inheriting TRL's own default of **0.0 -- no KL constraint at
  all**. Found while reviewing `../2026-04-24_nemotron-v1/`, which
  explicitly used `beta=0.01`. Given this project's repeated real GRPO NaN
  collapses (job 37933496 at step 12, others), an unconstrained policy is
  a plausible contributing factor worth controlling for directly. Now a
  documented, explicit `--kl_beta` argument, default `0.01` -- not yet
  validated on this backbone/recipe; the next Phase 2 run is the first
  test of whether this changes training stability.
- **R_tom: a new Theory of Mind reward component** (`core/rewards/theory_of_mind.py`,
  wired into `core/rewards/composite.py` as `R_tom` at weight `w_tom=0.5`,
  and into `training/grpo.py` as `make_tom_reward_func()` in the real GRPO
  reward list). Real gap found: `R_emp` (`core/rewards/empathy.py`) only
  ever scored `<user_belief>` on writing STYLE (word count, jargon
  substitution, Flesch-Kincaid readability) -- never on whether it's an
  ACCURATE model of what the recipient actually already knows. A response
  could be perfectly readable and completely wrong about the recipient's
  knowledge state and score identically to one that gets it right. R_tom
  checks `<user_belief>` against synthetic ground-truth
  `recipient_knows`/`recipient_does_not_know` facts (does it avoid
  redundantly re-explaining what's already known; does it correctly
  address what isn't) -- same embedding-similarity-fallback pattern
  already used by `R_retention`. Ground truth is sampled, not
  clinically real (clinicians always "know" the diagnosis/raw numbers;
  patients/family have a randomized 50/50 split on the diagnosis;
  nobody "knows" the prognosis) -- same honesty level as
  `core/cohort/grpo_prompts.py`'s existing `recipient_type` sampling,
  and explicitly flagged UNVALIDATED in every relevant docstring, same
  caveat as `R_meta`. Wired into BOTH data sources: Phase 1's synthetic
  vignette generator (`data/synthetic/vignette_generation.py`) and Phase
  2's real MIMIC-IV-derived prompt builder (`core/cohort/grpo_prompts.py`).
  Backward compatible -- older prompt datasets without the new columns
  default to `None`, which `reward_theory_of_mind` treats as "no ground
  truth" (neutral 1.0 score), not a crash or corrupted signal. 6 new
  tests, 118/118 total passing.
- **Closed a real test-coverage gap**: `reward_semantic_fidelity`,
  `reward_context_retention`, `reward_metacognitive_selfcorrection`, and
  `compute_total_reward` (the actual aggregate combining all eleven
  reward components) had ZERO direct test coverage in this repo --
  `test_reward_logic.py`'s own docstring referenced a
  `tests/test_reward_semantic.py` file that never actually existed here
  (confirmed via repo-wide grep, 2026-08-13). Wrote 19 new tests
  (`tests/test_rewards/test_reward_semantic.py`), every embedding-
  similarity threshold backed by real measured cosine similarities from
  the actual sentence encoder, not guessed. 137/137 total passing.
- **Fixed real, previously-undetected breakage in `scripts/streamlit_app.py`**:
  the file had never been run since the 2026-08-12 restructure. Every
  import referenced deleted modules (`models.multi_stream`,
  `hypergraph.verification`, `agents.tool_use`, `training.rewards`);
  `MODEL_REGISTRY` pointed at checkpoints that don't exist in this repo;
  the Reward Inspector tab had no UI fields for R_tom's
  `recipient_knows`/`recipient_does_not_know`, so R_tom could never
  actually be inspected there (always silently defaulted to neutral).
  All fixed; verified via a real headless boot test (SLURM job 39286154,
  `streamlit run --headless` + curl) -- confirmed the server boots and
  renders without a server-side exception. This proves the server-level
  fix works; it does NOT prove every tab/button's interactive behavior is
  correct, since Streamlit renders client-side over WebSocket and no
  browser is available in this environment to click through it -- an
  honest limit on what "validated" means here, not glossed over.
- **Exhausted the last remaining Phase 1 root-cause lead**: compared the
  real original v2 training job (37504804) against recent runs' GPU
  hardware/software banners -- identical (NVIDIA L4, Torch 2.13.0+cu130,
  CUDA 8.9, Toolkit 13.0) across all of them, on different but
  same-generation nodes. No driver-version logging exists to compare
  further, and no cluster-admin access is available to dig deeper. The
  v2-vs-v3+ format-compliance gap's root cause is now formally
  unresolved, not merely deprioritized -- every checkable hypothesis has
  been checked.

## Expected outputs, with real examples

The pipeline produces four kinds of artifact, one per phase, plus a
consolidated evaluation report. All examples below are real (pulled from
actual files in this project), not illustrative/hypothetical.

### 1. Phase 1/2 output: a raw four-stream generation

The model is trained to always answer inside four XML-style tags —
`<think>`, `<patient_state>`, `<forecast>`, `<user_belief>` — enforced by
`STREAM_SYSTEM_PROMPT` (`core/schema.py`) and checked by `core/parsing.py`.
A real generation from `checkpoints/phase1_sft_v3` (diagnostic sample):

```
PROMPT: A 68-year-old post-op patient has SBP 85, lactate 3.8, HR 110.
        What is your assessment?

<think>
SBP 85, lactate 3.8, and HR 110 together suggest early septic or
hypovolemic shock rather than isolated pain-related tachycardia in a
post-op patient...
</think>
<patient_state>
Hypotensive (SBP 85), tachycardic (HR 110), rising lactate (3.8) --
consistent with early shock of undetermined source; recommend urgent
reassessment and repeat labs within 1-2 hours.
</patient_state>
<forecast>
MAP_6h: 58 [52-64]
lactate_6h: 3.1 [2.4-3.9]
</forecast>
<user_belief>
Addressed to the covering surgical resident; keep reasoning clinically
concrete and flag the urgency without over-alarming.
</user_belief>
```

Every evaluation metric (format compliance, deterioration AUROC,
structural empathy) is computed by parsing text exactly like this. The
current baseline checkpoint (`phase1_sft_v2_reproduction`) only produces
this cleanly ~47% of the time (n=300) — the rest either omit a tag, get
the order wrong, or fail to close tags at all and run on until
`max_new_tokens` (confirmed via live diagnostic samples during the Phase
1 investigation).

### 2. Phase 3 output: a hyperedge in the derived hypergraph

From `hypergraph/derived_hypergraph_full_38777889.json` (4187 entries,
status `PENDING_CLINICAL_REVIEW`):

```json
{
  "variables": ["heart_rate_high", "sbp_low"],
  "support": 53515,
  "p_value": 0.0,
  "odds_ratio": 1.424,
  "observed_rate": 0.0616,
  "expected_rate_under_independence": 0.0441
}
```

Reading this: across the MIMIC-IV derivation cohort, "high heart rate"
and "low SBP" co-occur about 42% more often than chance would predict,
statistically significant (p=0.0), observed together in 53,515 time
windows. This is what `R_bound` (`core/rewards/boundary.py`) checks the
model's `<patient_state>` claims against during GRPO — and what a
clinician has to approve or reject one-by-one via
`scripts/clinical_review_ui.py` before it's trusted for training or
deployment.

### 3. Evaluation report: the six-endpoint JSON

Every evaluation run (`scripts/run_evaluation.sh` or the smoke-test
pattern) produces a report like this real one
(`evaluation/reports/phase1_sft_v2_reproduction_evaluation.json`, n=300):

```json
{
  "topological_fidelity": 0.927,
  "deterioration_auroc": 0.576,
  "deterioration_auprc": 0.420,
  "structural_empathy_mean": 0.636,
  "format_compliance_rate": 0.467,
  "deployment": {
    "schema_validity_rate": 0.467,
    "tool_call_success_rate": 1.0,
    "latency_p50_ms": 73852.8,
    "latency_p95_ms": 140690.3
  },
  "n_examples": 300,
  "deterioration_detection": {
    "auroc": 0.576, "precision": 0.370, "recall": 1.0, "f2_score": 0.746
  },
  "f2_optimal_threshold": null,
  "f2_at_optimal_threshold": null
}
```

(The last two fields — `f2_optimal_threshold` / `f2_at_optimal_threshold`
— are the F2-sweep addition described above; they'll be populated on the
next real evaluation run, since the source-repo report predates that
addition.) A `.png` and `.docx` visual report are also generated
alongside the JSON.

### 4. Phase 4 output: rollout tuples (not yet run)

`training/rollout.py`'s worker pool is built to push
`(prompt, completion, reward_components)` tuples onto a shared queue,
consumed in batches by the GRPO training loop. This phase has not yet
been exercised against real GPU/data, so no real sample exists yet —
only the schema it's built to produce.

### 5. User-facing output: the chatbot UI

Unlike the first four (files/artifacts), this one is the rendering layer
meant to actually be looked at by a person in real time —
`scripts/streamlit_app.py`. It takes a raw four-stream generation (see
example 1 above), parses it, and displays it as a normal chat response
rather than raw tagged text:

- **`💬 Response`** — the extracted substantive answer. Not simply
  `<user_belief>` (a real bug was caught and fixed here: `<user_belief>`
  is a reader-framing note, not itself an answer — showing it as the
  whole response meant a correctly-answered question could display
  nothing but "Assume a clinician-level reader..." with the real answer
  invisible, buried in `<think>`). `_extract_answer_content()` instead
  picks the actual substantive content from whichever stream has it
  (`<patient_state>` when applicable, otherwise an explicit "Answer:"
  marker inside `<think>`, otherwise the closing sentence as a fallback).
- **Side panel, always visible (not click-to-expand)** — `🧠 Reasoning`
  (`<think>`), `📋 Patient State`, `📈 Forecast` (parsed numeric
  predictions, e.g. `MAP_6h: 58 [52-64]`, or "Not applicable").
- **`🎯 Framed for: ...`** caption — the model's `<user_belief>` estimate
  of the reader, shown under the response.

For the real example in section 1 above, the chat would show:

> **💬 Response**
> Hypotensive (SBP 85), tachycardic (HR 110), rising lactate (3.8) --
> consistent with early shock of undetermined source; recommend urgent
> reassessment and repeat labs within 1-2 hours.
> 🎯 *Framed for: Addressed to the covering surgical resident; keep
> reasoning clinically concrete and flag the urgency without
> over-alarming.*

with the reasoning/patient-state/forecast breakdown visible alongside it,
not hidden.

If the model produces malformed output (missing/misordered tags — which
happens on roughly half of generations at the current baseline, per
`format_compliance_rate` above), the UI falls back to a warning plus the
raw text rather than silently breaking or hiding the failure.

## Repository status

This directory (`Agentic-Metacognitive-MDT_V1-August2026/`) is a
structural reorganization of `../Agentic-DT_V1-July/`, ported 2026-08-12.
See `README.md` for the layout rationale. **109/109 tests passed at the
initial port; 137/137 pass as of the latest addition** (see "Recent
additions" above for the full history: F2-threshold sweep, R_tom reward,
and the semantic/retention/metacognitive/compute_total_reward coverage
gap closure each added tests on top of the original port). Going forward,
this is the working directory for code changes; `../Agentic-DT_V1-July/`
remains the historical record of the investigation (git history, dated
logs) but is no longer where new work happens unless stated otherwise.
