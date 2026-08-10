#!/bin/bash
#SBATCH --job-name=ssfm-cifar
#SBATCH --partition=a6000
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=SSFM-jax/slurm/%x_%j.out

set -euo pipefail

CONDA_ENV="${SSFM_CONDA_ENV:-ssfm-cifar}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/SSFM-jax/ssfm_cifar.py" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/SSFM-jax"
    elif [[ -f "${SLURM_SUBMIT_DIR}/ssfm_cifar.py" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    else
        echo "Cannot locate SSFM-jax from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
        exit 1
    fi
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
