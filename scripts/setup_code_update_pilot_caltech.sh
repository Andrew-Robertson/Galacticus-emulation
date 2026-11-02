#!/bin/bash
# Run from an existing Galacticus build/runtime environment on Caltech.
set -euo pipefail
PILOT=$(cd "$(dirname "$0")" && pwd)
: "${GALACTICUS_EXEC_PATH:?Activate your usual Galacticus build environment first}"
: "${GALACTICUS_DATA_PATH:?Set the existing Galacticus datasets directory}"
: "${SHMR_PATH:?Set the existing make-SHMR-datasets directory}"
: "${GALACTICUS_EMU_ROOT:?Set the existing Galacticus-emulation repository directory}"
REVISION=$(python "$PILOT/pilot.py" pin-merged-master "$PILOT")
echo "Pinned Galacticus master: $REVISION (includes PRs #1499, #1665 and #1682)"
if [[ -e "$PILOT/source" ]]; then
  test "$(git -C "$PILOT/source" rev-parse HEAD)" = "$REVISION"
  git -C "$PILOT/source" diff --exit-code HEAD
else
  OLD_SOURCE=$GALACTICUS_EXEC_PATH
  if [[ -f "$OLD_SOURCE" ]]; then OLD_SOURCE=$(dirname "$OLD_SOURCE"); fi
  git -C "$OLD_SOURCE" rev-parse --show-toplevel
  git -C "$OLD_SOURCE" fetch https://github.com/galacticusorg/galacticus.git master
  git -C "$OLD_SOURCE" worktree add --detach "$PILOT/source" "$REVISION"
fi
git -C "$PILOT/source" submodule update --init --recursive
echo "Checkout ready: $PILOT/source"
echo 'Next: compile Galacticus.exe yourself on an interactive compute node, then run:'
printf 'python %q prepare %q --source %q\n' "$PILOT/pilot.py" "$PILOT" "$PILOT/source"
