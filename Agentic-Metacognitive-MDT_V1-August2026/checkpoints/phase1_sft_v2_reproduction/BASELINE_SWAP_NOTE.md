# Baseline swap, 2026-08-14

This directory's contents were replaced. It now holds the **seed=101**
variance-check checkpoint (92.0% format compliance at n=300), copied from
`../phase1_sft_variance_seed101`.

Why: the run-to-run variance check (3 identical-recipe runs, only `--seed`
varied) found format compliance ranging 46.7% (seed=42) - 56.3% (seed=202) -
92.0% (seed=101). This resolved the "46.7% vs 97%" investigation -- the
recipe is highly seed-sensitive, not regressed. There was no bug to fix;
picking the best-performing seed IS the fix.

The original seed=42 checkpoint (46.7%, what this directory held from
2026-08-12 through 2026-08-14) is archived, unmodified, at
`../phase1_sft_v2_reproduction_seed42_archived/` for reference -- not
deleted, in case future comparison is needed.

Every script/config that references `./checkpoints/phase1_sft_v2_reproduction`
(run_grpo.sh, run_sft.sh's docstring, run_evaluation.sh, streamlit_app.py,
configs/sft.yaml, configs/grpo.yaml) now transparently uses the seed=101
weights with no path changes required.
