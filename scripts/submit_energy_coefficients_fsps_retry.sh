#!/bin/bash
# Written by OpenAI Codex at Andrew Robertson's request.
set -euo pipefail
SETUP=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ENERGY_COEFFICIENT_OUTPUT:?Set a fresh campaign output directory}"
SOURCE=${GALACTICUS_EXEC_PATH:-/home/arobert2/Galacticus/galacticus}
EXE=${ENERGY_COEFFICIENT_EXECUTABLE:-$SOURCE/Galacticus.exe_energyCoefficients}
export GALACTICUS_DATA_PATH=${GALACTICUS_DATA_PATH:-/home/arobert2/Galacticus/datasets}
export SHMR_PATH=${SHMR_PATH:-/home/arobert2/GalacticusEmu/make-SHMR-datasets}
SHARED_TOOLS=${GALACTICUS_TOOLS_PATH:-${GALACTICUS_DATA_PATH%/}/dynamic}
PYTHON=$(command -v python)
"$PYTHON" -c 'import numpy, h5py, matplotlib, lxml.etree'
command -v "${FCCOMPILER:-gfortran}"
test -x "$EXE"
test -d "$SHARED_TOOLS"
test -d "$SHMR_PATH"
test ! -e "$ENERGY_COEFFICIENT_OUTPUT"
test ! -L "$ENERGY_COEFFICIENT_OUTPUT"
test ! -e "${ENERGY_COEFFICIENT_OUTPUT}_submission"
CONTROL="${ENERGY_COEFFICIENT_OUTPUT}_fsps_build"
export GALACTICUS_TOOLS_PATH="${ENERGY_COEFFICIENT_OUTPUT}_tools"
test ! -e "$GALACTICUS_TOOLS_PATH"
test ! -L "$GALACTICUS_TOOLS_PATH"
mkdir -p "$(dirname "$CONTROL")"
mkdir "$CONTROL"
cp "$SETUP/scripts/bootstrap_energy_fsps.py" "$CONTROL/bootstrap_energy_fsps.py"
cat > "$CONTROL/build.sh" <<'EOF'
#!/bin/bash
set -euo pipefail
exec "$1" "$2/bootstrap_energy_fsps.py" --shared-tools "$3" --output "$4"
EOF
BUILD=$(sbatch --parsable --partition=expansion --qos=normal --nodes=1 --ntasks=1 \
    --cpus-per-task=1 --mem=16G --time=02:00:00 --export=ALL --chdir="$SETUP" \
    --job-name=energy-fsps-build --output="$CONTROL/build-%j.out" --error="$CONTROL/build-%j.err" \
    "$CONTROL/build.sh" "$PYTHON" "$CONTROL" "$SHARED_TOOLS" "$GALACTICUS_TOOLS_PATH")
BUILD=${BUILD%%;*}
printf '%s\n' "$BUILD" > "$CONTROL/build_job_id.txt"
printf 'Private FSPS build: %s\nBuild logs: %s\n' "$BUILD" "$CONTROL"
bash "$SETUP/scripts/submit_energy_coefficients_caltech.sh" "$BUILD"
