"""
tests/conftest.py

PORTED 2026-08-12 from ../Agentic-DT_V1-July/tests/conftest.py. The
`requires_torch` marker registration and the `well_formed_generation` fixture
are unchanged. Added: a sys.path bootstrap so the tests import the package
from the repo root without needing an editable install, replacing the
per-file `sys.path.append(...)` lines every source test used to carry.

Shared pytest fixtures and configuration. Marks tests that require torch
(and therefore a GPU-capable environment, or at least a working torch
install) as `@pytest.mark.requires_torch`, so the torch-independent tests
(regex parsing, statistics, hyperedge mining logic) can be run anywhere --
including environments without GPU access -- via:

    pytest -m "not requires_torch"

while the full suite (including model-loading tests) is intended to run on
HiPerGator or another environment with the full training requirements installed:

    pytest
"""

import sys
from pathlib import Path

import pytest

# Repo root on sys.path so `import core`, `import training`, `import
# evaluation`, `import sandbox` resolve when running pytest from anywhere.
# The source repo did this with an identical sys.path.append line repeated at
# the top of all 13 test files; centralizing it here is the one structural
# change made to the test suite during the port.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_torch: marks tests that need torch/transformers installed (deselect with '-m \"not requires_torch\"')"
    )


@pytest.fixture
def well_formed_generation():
    """A single, reusable well-formed 4-stream generation string used across
    multiple test files, so the "canonical valid example" only needs to be
    defined and kept in sync with STREAM_TAGS in one place.
    """
    return (
        "<think>HR climbing, MAP dropping, wait, actually the lactate trend is more "
        "concerning here.</think>"
        "<patient_state>Tachycardic at 118, MAP 58, lactate rising to 3.1 -- concerning "
        "for early septic shock.</patient_state>"
        "<forecast>MAP_6h: 55 [48-62]\nlactate_6h: 3.8 [3.0-4.6]</forecast>"
        "<user_belief>This will be read by the bedside nurse; keep it concrete and "
        "action-oriented.</user_belief>"
    )
