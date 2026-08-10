#!/bin/bash
#SBATCH --job-name=ssfm-torch
#SBATCH --partition=a6000
#SBATCH --gres=gpu:2
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=SSFM-torch/slurm/%x_%j.out

set -euo pipefail
CONDA_ENV="${SSFM_CONDA_ENV:-ssfm-torch}"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    if [[ -f "${SLURM_SUBMIT_DIR}/SSFM-torch/train.py" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}/SSFM-torch"
    elif [[ -f "${SLURM_SUBMIT_DIR}/train.py" ]]; then
        SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
    else
        echo "Cannot locate SSFM-torch from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
        exit 1
    fi
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
fi
PYTHON_FILE="${1:-train.py}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "${PYTHON_FILE}" = /* ]]; then
    PYTHON_PATH="${PYTHON_FILE}"
else
    PYTHON_PATH="${SCRIPT_DIR}/${PYTHON_FILE}"
fi
if [[ ! -f "${PYTHON_PATH}" ]]; then
    echo "Python file not found: ${PYTHON_PATH}" >&2
    exit 1
fi
cd "${SCRIPT_DIR}"
if [[ "$(basename -- "${PYTHON_PATH}")" == "train.py" ]]; then
    srun conda run --no-capture-output -n "${CONDA_ENV}" \
        torchrun --standalone --nproc-per-node=gpu "${PYTHON_PATH}" "$@"
else
    srun conda run --no-capture-output -n "${CONDA_ENV}" python "${PYTHON_PATH}" "$@"
fi
