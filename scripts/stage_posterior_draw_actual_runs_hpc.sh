#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if compgen -G "${SCRIPT_DIR}/posterior_draw_[0-9][0-9]" > /dev/null; then
  RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}}"
fi

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "Set RUN_ROOT to the posteriorDrawRuns directory to stage." >&2
  exit 2
fi

STAGE_ROOT="${STAGE_ROOT:-$(dirname "${RUN_ROOT}")/posteriorDrawRuns_reduced_export}"
MAKE_TARBALL="${MAKE_TARBALL:-0}"

if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Missing RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi

rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_ROOT}"

if [[ -f "${RUN_ROOT}/posterior_draw_manifest.csv" ]]; then
  cp -p "${RUN_ROOT}/posterior_draw_manifest.csv" "${STAGE_ROOT}/"
fi

if [[ -d "${RUN_ROOT}/logs" ]]; then
  mkdir -p "${STAGE_ROOT}/logs"
  find "${RUN_ROOT}/logs" -maxdepth 1 -type f \( -name '*.out' -o -name '*.err' \) -exec cp -p {} "${STAGE_ROOT}/logs/" \;
fi

for run_dir in "${RUN_ROOT}"/posterior_draw_[0-9][0-9]; do
  [[ -d "${run_dir}" ]] || continue
  draw_label="$(basename "${run_dir}")"
  out_dir="${STAGE_ROOT}/${draw_label}"
  mkdir -p "${out_dir}"

  for file_name in \
    romanEPS_massFunction_reduced.hdf5 \
    model_changes.xml \
    params.xml; do
    if [[ -f "${run_dir}/${file_name}" ]]; then
      cp -p "${run_dir}/${file_name}" "${out_dir}/"
    else
      echo "Warning: missing ${draw_label}/${file_name}" >&2
    fi
  done

  if [[ -d "${run_dir}/emission_line_dust" ]]; then
    cp -a "${run_dir}/emission_line_dust" "${out_dir}/"
  else
    echo "Warning: missing ${draw_label}/emission_line_dust" >&2
  fi
done

{
  echo "# Posterior draw reduced export"
  echo "# Source: ${RUN_ROOT}"
  echo "# Staged: ${STAGE_ROOT}"
  echo
  find "${STAGE_ROOT}" -type f | sort
} > "${STAGE_ROOT}/file_inventory.txt"

(
  cd "${STAGE_ROOT}"
  find . -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS.txt
)

du -sh "${STAGE_ROOT}"
echo "Staged posterior draw files in: ${STAGE_ROOT}"

if [[ "${MAKE_TARBALL}" == "1" ]]; then
  tarball="${STAGE_ROOT}.tar.gz"
  tar -C "$(dirname "${STAGE_ROOT}")" -czf "${tarball}" "$(basename "${STAGE_ROOT}")"
  du -sh "${tarball}"
  echo "Created tarball: ${tarball}"
fi
