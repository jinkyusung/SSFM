#!/bin/bash
#SBATCH --job-name=ssfm-cifar
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=./slurm/%x_%A_%a.out

set -euo pipefail

CONDA_ENV="${SSFM_CONDA_ENV:-ssfm-cifar}"
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
    echo "Create it first with: ./setup_ssfm_cifar_env.sh cuda12 ${CONDA_ENV}" >&2
    exit 1
fi

if [[ ! -f "${PYTHON_FILE}" ]]; then
    echo "Python file not found: ${PYTHON_FILE}" >&2
    echo "Usage: sbatch run.sh [file.py] [python arguments...]" >&2
    exit 1
fi

mkdir -p slurm
srun conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${PYTHON_FILE}" "$@"
