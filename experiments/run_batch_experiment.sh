#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

IMAGES_PATH="${IMAGES_PATH:-data/MNIST/raw/t10k-images-idx3-ubyte}"
LABELS_PATH="${LABELS_PATH:-data/MNIST/raw/t10k-labels-idx1-ubyte}"
COUNT="${COUNT:-100}"
START="${START:-0}"
PROGRESS="${PROGRESS:-50}"
RUN_NAME="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"

RUN_DIR="results/experiments/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

ARGS="--images ${IMAGES_PATH} --labels ${LABELS_PATH} --outdir ${RUN_DIR} --count ${COUNT} --start ${START} --progress ${PROGRESS} --quiet +quiet"
make run_batch ARGS="${ARGS}"

INDEX_FILE="results/experiments/index.csv"
if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "run_name,timestamp_utc,count,start,outdir" > "${INDEX_FILE}"
fi

echo "${RUN_NAME},$(date -u +%Y-%m-%dT%H:%M:%SZ),${COUNT},${START},${RUN_DIR}" >> "${INDEX_FILE}"
echo "Experiment output: ${RUN_DIR}"
