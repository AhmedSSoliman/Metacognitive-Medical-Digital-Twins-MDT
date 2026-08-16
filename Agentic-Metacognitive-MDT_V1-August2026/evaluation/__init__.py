"""
evaluation

The proposal's evaluation endpoints. Needs torch only for INFERENCE (to
produce the generations being scored) -- the scoring functions themselves
depend on numpy/pandas/sklearn, plus sentence-transformers for the
embedding-based ones, and none of these modules imports torch at module level.

  topological.py     Endpoint 1: Topological Fidelity (reuses R_bound).
  predictive.py      Endpoint 2: AUROC/AUPRC/P/R/F2 + bootstrap CI;
                     also Endpoint 6's schema-validity and deployment metrics.
  retention.py       Endpoint 3: trajectory MAE (see its naming note).
  communication.py   Endpoint 5: structural empathy (reuses R_emp).
  delta_embedding.py R_meta audit vs expert annotation (reuses R_meta).
  report.py          EvaluationResult + run_full_evaluation + JSON output.
  smoke_test_cases.py Qualitative 6-case clinical battery (needs a model).
  ablation.py        Placeholder -- no ablation runner exists yet.

Endpoint 4 (HITL Audit Latency) has no code: it is measured by human
reviewers, not computed.
"""
