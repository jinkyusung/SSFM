#!/bin/bash
#SBATCH --job-name=ssfm-cifar
#SBATCH --partition=a6000
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null

set -euo pipefail

CONDA_ENV="${SSFM_CONDA_ENV:-ssfm-cifar}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLURM_LOG_DIR="${SCRIPT_DIR}/slurm"
mkdir -p "${SLURM_LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LOG_NAME="${SLURM_JOB_NAME:-ssfm-cifar}_${SLURM_JOB_ID}.out"
    exec >"${SLURM_LOG_DIR}/${LOG_NAME}" 2>&1
fi
PYTHON_FILE="${1:-ssfm_cifar.py}"
if [[ $# -gt 0 ]]; then
    shift
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available on the compute node PATH." >&2
    exit 1
fi

if ! conda run -n "${CONDA_ENV}" python --version >/dev/null 2>&1; then
    echo "Conda environment not found or unusable: ${CONDA_ENV}" >&2
    echo "Create it first with: ${SCRIPT_DIR}/setup_ssfm_cifar_env.sh cuda12 ${CONDA_ENV}" >&2
    exit 1
fi

if [[ "${PYTHON_FILE}" = /* ]]; then
    PYTHON_PATH="${PYTHON_FILE}"
else
    PYTHON_PATH="${SCRIPT_DIR}/${PYTHON_FILE}"
fi

if [[ ! -f "${PYTHON_PATH}" ]]; then
    echo "Python file not found: ${PYTHON_PATH}" >&2
    echo "Usage: sbatch run.sh [file.py] [python arguments...]" >&2
    exit 1
fi

cd "${SCRIPT_DIR}"
srun conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${PYTHON_PATH}" "$@"
