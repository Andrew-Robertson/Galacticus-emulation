#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <campaign_root> [output_suffix]"
  exit 1
fi

CAMPAIGN_ROOT="$1"
OUTPUT_SUFFIX="${2:-_full18_opt}"

cd "$(dirname "$0")/.."

echo "Refreshing Trinity summary table..."
python scripts/extract_trinity_summary.py "${CAMPAIGN_ROOT}"

echo "Rebuilding merged emulator table..."
python - <<'PY' "${CAMPAIGN_ROOT}"
import sys
from pathlib import Path
import pandas as pd

campaign = Path(sys.argv[1]).resolve()
samples = pd.read_csv(campaign / "samples.csv")
summary = pd.read_csv(campaign / "summary.csv")
merged = samples.merge(summary, on="evaluation_id", how="inner", validate="one_to_one")
merged.to_csv(campaign / "emulator_table.csv", index=False)
print(campaign / "emulator_table.csv")
print(f"rows={len(merged)}")
PY

echo "Fitting optimized full-18-output GP CV..."
python scripts/fit_trinity_gp.py \
  "${CAMPAIGN_ROOT}" \
  --output-suffix "${OUTPUT_SUFFIX}" \
  --optimize-hyperparameters \
  --n-restarts-optimizer 1

echo "Running optimized subset-size comparison..."
python scripts/compare_trinity_gp_subset_sizes.py \
  "${CAMPAIGN_ROOT}" \
  --sizes 32 64 128 256 512 \
  --output-suffix "${OUTPUT_SUFFIX}" \
  --optimize-hyperparameters \
  --n-restarts-optimizer 1

echo "Done."
