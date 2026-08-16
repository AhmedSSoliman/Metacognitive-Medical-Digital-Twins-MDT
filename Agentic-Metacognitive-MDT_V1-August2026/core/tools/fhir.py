"""
core/tools/fhir.py

PLACEHOLDER -- FHIR-compatible summary generation is NOT YET IMPLEMENTED.

Status as of this port (2026-08-12): there is NO FHIR code anywhere in the
source repo (../Agentic-DT_V1-July/). This was verified by a case-insensitive
grep for 'fhir' across the entire source tree, which returned zero matches in
any file. Nothing was dropped, moved, or renamed in this port -- this
capability simply does not exist yet.

Intended future scope, per the target architecture: render a generated
<patient_state> (and optionally the <forecast> stream) into a FHIR-compatible
resource bundle -- most plausibly an Observation / RiskAssessment /
Composition set -- so a summary produced by this system can be written back
into a real EHR rather than only read as free text.

What exists today instead: core/tools/dispatch.py implements the ReAct-style
tool-call parsing and execution side (ToolRegistry, extract_tool_calls,
make_default_registry with query_hypergraph and get_recent_labs). A FHIR
exporter would most naturally be registered as an additional tool there, or
called after generation on the parsed streams from core/parsing.py.
"""

from __future__ import annotations
