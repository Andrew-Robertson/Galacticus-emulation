#!/bin/bash
# Written by OpenAI Codex at Andrew Robertson's request.
# Submit preparation after a successful build, then a matched 12-case array.
set -euo pipefail
SETUP=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_JOB=${1:?Pass a build job ID, or built to reuse a completed executable}
[[ "$BUILD_JOB" == built || "$BUILD_JOB" =~ ^[0-9]+$ ]]
OUT=${ENERGY_COEFFICIENT_OUTPUT:-/resnick/groups/carnegie_poc/arobert2/galacticusEmulationCampaigns/equilibriumSizesEnergy/coefficient_sweep_20261124}
SOURCE=${GALACTICUS_EXEC_PATH:-/home/arobert2/Galacticus/galacticus}
EXE=${ENERGY_COEFFICIENT_EXECUTABLE:-$SOURCE/Galacticus.exe_energyCoefficients}
BUILD_DEPENDENCY=''
if [[ "$BUILD_JOB" == built ]]; then
    test -x "$EXE"
else
    BUILD_DEPENDENCY="--dependency=afterok:$BUILD_JOB"
fi
export GALACTICUS_DATA_PATH=${GALACTICUS_DATA_PATH:-/home/arobert2/Galacticus/datasets}
export SHMR_PATH=${SHMR_PATH:-/home/arobert2/GalacticusEmu/make-SHMR-datasets}
PYTHON=$(command -v python)
"$PYTHON" -c 'import numpy, h5py, matplotlib, lxml.etree'
test -d "$GALACTICUS_DATA_PATH"
test -d "$SHMR_PATH"
test ! -e "$OUT"
test ! -L "$OUT"
CONTROL="${OUT}_submission"
mkdir -p "$(dirname "$CONTROL")"
mkdir "$CONTROL"

cat > "$CONTROL/prepare.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
exec "$1" "$2/scripts/energy_coefficient_sweep.py" prepare --root "$3" --source "$4" --executable "$5"
EOF
cat > "$CONTROL/run.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
exec "$1" "$2/pilot.py" run "$2" --task "$SLURM_ARRAY_TASK_ID"
EOF
cat > "$CONTROL/analyze.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
exec "$1" "$2/analyze_energy_precision_pilot.py" "$2" --output "${2}_reduced"
EOF

COMMON=(--parsable --partition=expansion --qos=normal --nodes=1 --ntasks=1 --export=ALL --chdir="$SETUP" --kill-on-invalid-dep=yes)
PREP=$(sbatch "${COMMON[@]}" --job-name=energy-coeff-prepare --cpus-per-task=8 --mem=64G --time=04:00:00 \
    ${BUILD_DEPENDENCY:+"$BUILD_DEPENDENCY"} --output="$CONTROL/prepare-%j.out" --error="$CONTROL/prepare-%j.err" \
    "$CONTROL/prepare.sh" "$PYTHON" "$SETUP" "$OUT" "$SOURCE" "$EXE")
PREP=${PREP%%;*}
printf '%s\n' "$PREP" > "$CONTROL/prepare_job_id.txt"
RUN=$(sbatch "${COMMON[@]}" --job-name=energy-coefficients --cpus-per-task=8 --mem=64G --time=72:00:00 \
    --array="0-11%${ENERGY_COEFFICIENT_CONCURRENCY:-4}" --dependency="afterok:$PREP" \
    --output="$CONTROL/run-%A_%a.out" --error="$CONTROL/run-%A_%a.err" \
    "$CONTROL/run.sh" "$PYTHON" "$OUT")
RUN=${RUN%%;*}
printf '%s\n' "$RUN" > "$CONTROL/run_job_id.txt"
ANALYSIS=$(sbatch "${COMMON[@]}" --job-name=energy-coeff-analysis --cpus-per-task=1 --mem=32G --time=08:00:00 \
    --dependency="afterok:$PREP,afterany:$RUN" --output="$CONTROL/analysis-%j.out" --error="$CONTROL/analysis-%j.err" \
    "$CONTROL/analyze.sh" "$PYTHON" "$OUT")
printf '%s\n' "${ANALYSIS%%;*}" > "$CONTROL/analysis_job_id.txt"
printf 'Preparation: %s\nArray: %s\nAnalysis: %s\nLogs/job IDs: %s\nCopy back: %s_reduced\n' \
    "$PREP" "$RUN" "${ANALYSIS%%;*}" "$CONTROL" "$OUT"
