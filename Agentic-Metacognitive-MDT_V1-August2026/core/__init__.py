"""
core

Torch-free foundation of the MDT system: stream schema and parsing, the
reward components, hypergraph construction/verification, MIMIC-IV cohort
handling, and tool dispatch.

DEPENDENCY BOUNDARY (the organizing principle of this package): nothing under
core/ imports torch, transformers, trl, unsloth, or peft AT MODULE IMPORT
TIME. The three embedding-based reward components need sentence-transformers
(and therefore torch) when CALLED, but they import it lazily inside the
function body -- see core/rewards/_encoder.py. Everything else here is regex,
string, numpy, pandas, scipy, or sklearn.

This is not stylistic. See core/schema.py's preserved docstring for the real
bug that motivated it: the parsing tests could not even be COLLECTED on a
machine without torch, because the pure-regex parser lived in a module that
imported torch at the top for unrelated model-loading classes.
"""
