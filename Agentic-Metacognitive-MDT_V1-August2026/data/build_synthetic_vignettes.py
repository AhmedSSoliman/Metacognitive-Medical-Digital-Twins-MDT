"""
data/build_synthetic_vignettes.py

Builds the expanded Phase 1 SFT synthetic vignette set, addressing two real
gaps found while testing a trained checkpoint (see
data/synthetic/vignette_generation.py's module docstring for the full
root-cause investigation):

1. All 36 original hand-written vignettes were missing a real `forecast`
   field -- backfilled here via vignette_generation.backfill_forecasts(),
   which infers the trending abnormal variable(s) from each vignette's own
   patient_state text and projects a clinically plausible 6h-ahead value.
   Vignettes with no detectable trending abnormality (genuine reassurance
   cases) correctly keep "not applicable" rather than having one
   fabricated.
2. Only 36 unique vignettes existed, requiring ~48x oversampling to reach
   an 8% Phase 1 training-mix fraction -- ~964 new, independently-sampled
   vignettes are generated across 25 clinical archetypes to reach ~1000
   total, cutting the required oversampling factor roughly 27-fold.

The original 36 are preserved unmodified at
data/synthetic/stream_format_vignettes_original36_backup.jsonl before this
script overwrites the main file, so no hand-curated content is lost.

Usage:
    python data/build_synthetic_vignettes.py
    python data/build_synthetic_vignettes.py --target_total 1500 --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.synthetic.vignette_generation import backfill_forecasts, generate_archetype_vignettes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--vignettes_path",
                         default=str(project_root / "data" / "synthetic" / "stream_format_vignettes.jsonl"))
    parser.add_argument("--backup_path",
                         default=str(project_root / "data" / "synthetic" /
                                     "stream_format_vignettes_original36_backup.jsonl"),
                         help="Where the original, unmodified 36 hand-written vignettes are preserved.")
    parser.add_argument("--target_total", type=int, default=1000,
                         help="Total vignette count after backfilling + generation.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    vignettes_path = Path(args.vignettes_path)
    original = []
    with open(vignettes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                original.append(json.loads(line))
    logger.info("Loaded %d original hand-written vignettes from %s", len(original), vignettes_path)

    shutil.copy(vignettes_path, args.backup_path)
    logger.info("Backed up original, unmodified vignettes to %s", args.backup_path)

    backfilled = backfill_forecasts(original, seed=args.seed)
    n_with_real_forecast = sum(1 for v in backfilled if v["forecast"].strip().lower() != "not applicable")
    logger.info("Backfilled forecasts: %d/%d original vignettes now have a real numeric forecast "
                "(the rest correctly remain 'not applicable' -- no detectable trending abnormality).",
                n_with_real_forecast, len(backfilled))

    n_to_generate = max(0, args.target_total - len(backfilled))
    generated = generate_archetype_vignettes(n_to_generate, seed=args.seed + 1)
    logger.info("Generated %d new archetype-based vignettes across %d clinical archetypes.",
                len(generated), len(set(v["_archetype"] for v in generated)))

    combined = backfilled + generated
    rng = random.Random(args.seed + 2)
    rng.shuffle(combined)  # interleave hand-written and generated, and across archetypes

    with open(vignettes_path, "w") as f:
        for v in combined:
            f.write(json.dumps(v) + "\n")
    logger.info("Wrote %d total vignettes -> %s", len(combined), vignettes_path)

    n_real_forecast_total = sum(1 for v in combined if v["forecast"].strip().lower() != "not applicable")
    logger.info("Final set: %d/%d vignettes (%.0f%%) have a real numeric forecast, vs. 0/36 (0%%) before this fix.",
                n_real_forecast_total, len(combined), 100 * n_real_forecast_total / len(combined))


if __name__ == "__main__":
    main()
