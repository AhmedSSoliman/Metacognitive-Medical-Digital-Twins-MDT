# Dangerous trajectories data -- placeholder

No dangerous-trajectories dataset exists in the source repository
(`Agentic-DT_V1-July`). Confirmed by exhaustive search (grep for
`dangerous.?traj|trajector` and `find` for filenames matching
`*dangerous_trajector*`) across the entire source tree: no such data file
was found.

The closest existing concept is `InterimRuleBasedChecker` in
`hypergraph/verification.py`, which encodes a small, hand-specified set of
physiologically-grounded rules for detecting implausible/dangerous
forecasted transitions directly in Python -- it does not read from or
write a "dangerous trajectories" dataset on disk.

This directory is reserved for such a dataset (e.g. curated examples of
clinically dangerous state transitions used for evaluation or as negative
training/reward examples) should one be built in the future.
