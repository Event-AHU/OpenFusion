#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT="${SCRIPT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python3}
DEVICE=${DEVICE:-cuda:3}
RUN_NAME=${RUN_NAME:-05_full_pnot4}

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" HeatOperator/run_train.py \
  --config "${REPO_ROOT}/config.yaml" \
  --override "device=${DEVICE}" \
  --override "run_name=${RUN_NAME}" \
  "$@"
