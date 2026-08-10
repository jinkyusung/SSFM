#!/usr/bin/env bash
set -Eeuo pipefail

# Usage: ./setup_env.sh [auto|cuda12|cpu] [conda-environment]
BACKEND="${1:-auto}"
ENV_NAME="${2:-ssfm-torch}"
PYTHON_VERSION="3.11"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

command -v conda >/dev/null 2>&1 || die "conda is not installed or not on PATH."
if [[ "${BACKEND}" == "auto" ]]; then
    if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
        BACKEND="cuda12"
    else
        BACKEND="cpu"
    fi
fi
[[ "${BACKEND}" == "cuda12" || "${BACKEND}" == "cpu" ]] || die "backend must be auto, cuda12, or cpu"

if ! conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
if [[ "${BACKEND}" == "cuda12" ]]; then
    conda run -n "${ENV_NAME}" python -m pip install \
        --index-url https://download.pytorch.org/whl/cu126 torch torchvision
elif [[ "$(uname -s)" == "Linux" ]]; then
    conda run -n "${ENV_NAME}" python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu torch torchvision
else
    conda run -n "${ENV_NAME}" python -m pip install torch torchvision
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
conda run -n "${ENV_NAME}" python -m pip install -e "${SCRIPT_DIR}[test]"
conda run -n "${ENV_NAME}" python -m pytest "${SCRIPT_DIR}/tests"
printf 'Environment ready. Activate with: conda activate %s\n' "${ENV_NAME}"

