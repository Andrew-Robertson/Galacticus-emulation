#!/bin/bash
# Prepare and submit the practical pilot plus frozen tighter comparisons on OBS-HPC.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
SOURCE=${GALACTICUS_EXEC_PATH:-/home/arobertson/Galacticus/galacticus-binding-energy}
EXE=${ENERGY_PILOT_EXECUTABLE:-$SOURCE/Galacticus.exe}
ORIGINAL=${ENERGY_PILOT_ORIGINAL:-/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_eps}
OUT=${ENERGY_PILOT_OUTPUT:-/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_precision}
PYTHON=${ENERGY_PILOT_PYTHON:-python}
export GALACTICUS_DATA_PATH=${GALACTICUS_DATA_PATH:-/home/arobertson/Galacticus/datasets}
export SHMR_PATH=${SHMR_PATH:-/home/arobertson/Galacticus/make-SHMR-datasets}
command -v sbatch >/dev/null
"$PYTHON" -c 'import lxml.etree, numpy, h5py, matplotlib'
LINK="$REPO/runs/campaigns/equilibriumSizesEnergy/pilot_128_precision"
if [[ -e "$LINK" || -L "$LINK" ]]; then
  echo "Existing campaign link/directory: $LINK. Use a fresh campaign; nothing submitted." >&2
  exit 1
fi
"$PYTHON" scripts/prepare_energy_precision_pilot.py prepare \
  --original "$ORIGINAL" --source "$SOURCE" --executable "$EXE" --output "$OUT" \
  --partition obs --cpus "${ENERGY_PILOT_CPUS:-8}" \
  --concurrency "${ENERGY_PILOT_CONCURRENCY:-0}" --walltime "${ENERGY_PILOT_WALLTIME:-72:00:00}"
# Validate both precision levels before launching the array. All 160 generated
# files have already passed the same numerical/physical-coordinate checks.
"$PYTHON" "$OUT/pilot.py" validate "$OUT"
mkdir -p "$(dirname "$LINK")"
ln -s "$OUT" "$LINK"
RUN_JOB=$(sbatch --parsable "$OUT/submit.slurm")
RUN_JOB=${RUN_JOB%%;*}
printf '%s\n' "$RUN_JOB" > "$OUT/run_job_id.txt"
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterany:$RUN_JOB" "$OUT/analyze.slurm")
printf '%s\n' "${ANALYSIS_JOB%%;*}" > "$OUT/analysis_job_id.txt"
ln -s "${OUT}_reduced" "${LINK}_reduced"
printf 'Simulation array: %s\nAnalysis: %s\nRuns: %s\nCopy back after analysis: %s_reduced\n' \
  "$RUN_JOB" "$ANALYSIS_JOB" "$OUT" "$OUT"
