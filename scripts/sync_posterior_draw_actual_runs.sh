#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${REMOTE_HOST:-}" || -z "${REMOTE_ROOT:-}" || -z "${LOCAL_ROOT:-}" ]]; then
  echo "Usage: REMOTE_HOST=user@host REMOTE_ROOT=/path/to/posteriorDrawRuns LOCAL_ROOT=/path/to/local/output $0" >&2
  exit 2
fi

DRY_RUN_FLAG=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_FLAG=(--dry-run)
fi

mkdir -p "${LOCAL_ROOT}"

rsync -av "${DRY_RUN_FLAG[@]}" --prune-empty-dirs \
  --include 'posterior_draw_*/' \
  --include 'posterior_draw_*/romanEPS_massFunction_reduced.hdf5' \
  --include 'posterior_draw_*/model_changes.xml' \
  --include 'posterior_draw_*/params.xml' \
  --include 'posterior_draw_*/emission_line_dust/' \
  --include 'posterior_draw_*/emission_line_dust/***' \
  --exclude '*' \
  "${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${LOCAL_ROOT}/"
