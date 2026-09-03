#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 CAMPAIGN_ROOT [EXTRA_ARGS...]" >&2
  exit 1
fi

CAMPAIGN_ROOT="$1"
shift

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT="${SCRIPT_DIR}/fit_multid_function_holdout_demo.py"

PRESETS=(
  smf_zfourge_z0
  smf_zfourge_z3
  mzr_blanc2019
  hi_mass_function_alfalfa
  sfr_function_robotham2011
  size_mass_vdw2014_star_forming_z0
  size_mass_vdw2014_quiescent_z0
)

for PRESET in "${PRESETS[@]}"; do
  echo "============================================================"
  echo "Starting preset: ${PRESET}"
  date
  python -u "$SCRIPT" "$CAMPAIGN_ROOT" --preset "$PRESET" --skip-existing "$@"
  echo "Finished preset: ${PRESET}"
  date
done

echo "============================================================"
echo "All requested holdout demos completed."
date
