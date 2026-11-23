#!/bin/bash
# Prepare 64 practical + 16 tighter EPS models using Galacticus PR #1854.
# Written by OpenAI Codex at Andrew Robertson's request.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
REVISION=f6648df263bf65b582f44c29683b72ab3ad27db6
SOURCE=${GALACTICUS_EXEC_PATH:-/home/arobertson/Galacticus/galacticus-binding-energy}
EXE=${ENERGY_PILOT_EXECUTABLE:-$SOURCE/Galacticus.exe_iterationLimitPilot}
ORIGINAL=${ENERGY_PILOT_ORIGINAL:-/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_eps}
OUT=${ENERGY_PILOT_OUTPUT:-/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_64_iteration_limit_20261123}
MASK=${ENERGY_PILOT_SURVEY_MASK:-/carnegie/nobackup/users/arobertson/Galacticus/Galacticus-emulation/equilibriumSizesEnergy/pilot_128_solver_fixed_20261120/practical/evaluations/pilot_128-eval-0000/dynamic/surveyGeometry/SDSS/sdss_dr72safe0_res6d.pol}
PYTHON=${ENERGY_PILOT_PYTHON:-python}
export GALACTICUS_DATA_PATH=${GALACTICUS_DATA_PATH:-/home/arobertson/Galacticus/datasets}
export SHMR_PATH=${SHMR_PATH:-/home/arobertson/Galacticus/make-SHMR-datasets}
command -v sbatch >/dev/null
"$PYTHON" -c 'import lxml.etree, numpy, h5py, matplotlib'
"$PYTHON" scripts/prepare_energy_precision_pilot.py prepare \
  --original "$ORIGINAL" --source "$SOURCE" --executable "$EXE" --output "$OUT" \
  --expected-revision "$REVISION" --require-solver-fixes --isolated-cache \
  --n-practical 64 --n-tighter 16 --iteration-limit 1000 --survey-mask "$MASK" \
  --partition obs --cpus "${ENERGY_PILOT_CPUS:-8}" \
  --concurrency "${ENERGY_PILOT_CONCURRENCY:-8}" --walltime "${ENERGY_PILOT_WALLTIME:-72:00:00}"
"$PYTHON" "$OUT/pilot.py" validate "$OUT"
RUN_JOB=$(sbatch --parsable "$OUT/submit.slurm")
RUN_JOB=${RUN_JOB%%;*}
printf '%s\n' "$RUN_JOB" > "$OUT/run_job_id.txt"
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterany:$RUN_JOB" "$OUT/analyze.slurm")
printf '%s\n' "${ANALYSIS_JOB%%;*}" > "$OUT/analysis_job_id.txt"
printf 'Simulation array: %s\nAnalysis: %s\nRuns: %s\nCopy back: %s_reduced\n' \
  "$RUN_JOB" "$ANALYSIS_JOB" "$OUT" "$OUT"
