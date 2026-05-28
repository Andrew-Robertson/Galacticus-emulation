#!/bin/bash
#SBATCH --job-name=posteriorDraws
#SBATCH --partition=expansion
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=40:30:00
#SBATCH --array=0-24%4
#SBATCH --output=/home/arobert2/GalacticusEmu/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_freeYield_20p_moreHalos_512/posteriorDrawRuns/logs/slurm-%A_%a.out
#SBATCH --error=/home/arobert2/GalacticusEmu/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_freeYield_20p_moreHalos_512/posteriorDrawRuns/logs/slurm-%A_%a.err

set -euo pipefail

ulimit -c 0
export GFORTRAN_ERROR_DUMPCORE=NO
ulimit -t unlimited
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

source /resnick/groups/carnegie_poc/arobert2/miniconda3/etc/profile.d/conda.sh
conda activate galacticus-workspace

export GALACTICUS_EMU_ROOT="${GALACTICUS_EMU_ROOT:-/home/arobert2/GalacticusEmu/Galacticus-emulation}"

RUN_ROOT="${RUN_ROOT:-/home/arobert2/GalacticusEmu/campaigns/sobol_mass_function_emissionlines_dust_simpleSizes_freeYield_20p_moreHalos_512/posteriorDrawRuns}"
DRAW_XML_DIR="${DRAW_XML_DIR:-$SLURM_SUBMIT_DIR}"
DRAW_INDEX="${SLURM_ARRAY_TASK_ID}"
DRAW_LABEL=$(printf "posterior_draw_%02d" "$DRAW_INDEX")
DRAW_XML="$DRAW_XML_DIR/${DRAW_LABEL}.xml"
RUN_DIR="$RUN_ROOT/$DRAW_LABEL"
GALACTICUS_OUTPUT="romanEPS_massFunction.hdf5"

if [[ ! -f "$DRAW_XML" ]]; then
  echo "Missing draw XML: $DRAW_XML" >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$RUN_ROOT/logs"
cd "$RUN_DIR"

cp "$DRAW_XML" model_changes.xml
if [[ -f "$DRAW_XML_DIR/posterior_draw_manifest.csv" ]]; then
  cp "$DRAW_XML_DIR/posterior_draw_manifest.csv" "$RUN_ROOT/posterior_draw_manifest.csv"
fi

echo "Starting $DRAW_LABEL"
echo "XML source: $DRAW_XML"
echo "Run directory: $RUN_DIR"
date

"$GALACTICUS_EXEC_PATH/Galacticus.exe_fixBHMSigmaReport" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/romanEPS.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/massFunction.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/handCalibratedChanges.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/odeAcc1e-4.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/seeds/seed1234.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/additionalQuantities.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/universeMachine/Trinity/universeMachine_Trinity_z012_outputs.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/haloMasses/massTreeMax1e14.xml" \
  "$GALACTICUS_PARAMS_PATH/starFormationHistory/adaptiveSFH-noSave.xml" \
  "$GALACTICUS_PARAMS_PATH/fineTuningFiles/emissionLineLuminosityFunctions.xml" \
  "$GALACTICUS_PARAMS_PATH/emissionLines/emissionLines.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/sizes/galacticStructureSolver-simple.xml" \
  "$GALACTICUS_PARAMETER_FILES/hierarchicalParameters/fineTuningFiles/moreHalos.xml" \
  model_changes.xml \
  --output-processed-parameters params.xml

echo "Galacticus finished for $DRAW_LABEL, now post-processing emission-line LFs"
date

python "$GALACTICUS_EMU_ROOT/scripts/process_emission_line_dust_evaluation.py" \
  "$GALACTICUS_OUTPUT" \
  --evaluation-id "$DRAW_LABEL" \
  --evaluation-index "$DRAW_INDEX" \
  --map-changes-xml model_changes.xml \
  --fixed-dust-case-label "$DRAW_LABEL" \
  --scatter-mode expected \
  --output-dir emission_line_dust

if [[ "${MAKE_REDUCED_COPY:-1}" == "1" ]]; then
  echo "Creating reduced HDF5 copy for $DRAW_LABEL"
  python "$GALACTICUS_EMU_ROOT/scripts/extract_campaign_hdf5_groups.py" \
    "$GALACTICUS_OUTPUT" \
    --output-filename romanEPS_massFunction_reduced.hdf5 \
    --overwrite
fi

if [[ "${RUN_MASS_FUNCTION_PLOT:-1}" == "1" ]]; then
  echo "Generating mass-function diagnostic plots for $DRAW_LABEL"
  python "$GALACTICUS_DUST_ROOT/dust_model/scripts/plotMassFunctionResults.py"
fi

echo "Job completed successfully for $DRAW_LABEL"
date
