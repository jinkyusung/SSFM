#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./setup_ssfm_cifar_env.sh [auto|cuda12|cpu] [conda-env-name] [train|metrics]
#
# Examples:
#   ./setup_ssfm_cifar_env.sh                 # auto-detect backend
#   ./setup_ssfm_cifar_env.sh cuda12          # Linux + NVIDIA CUDA 12
#   ./setup_ssfm_cifar_env.sh cpu ssfm-cifar  # CPU/macOS
#   ./setup_ssfm_cifar_env.sh auto ssfm-cifar metrics  # include FID packages

BACKEND="${1:-auto}"
ENV_NAME="${2:-ssfm-cifar}"
INSTALL_PROFILE="${3:-train}"
PYTHON_VERSION="3.10"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${SCRIPT_DIR}/ssfm_cifar.py"

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

command -v conda >/dev/null 2>&1 || die "conda is not installed or not on PATH."
[[ -f "${MODEL_PATH}" ]] || die "Cannot find ${MODEL_PATH}."

case "${BACKEND}" in
    auto)
        if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
            BACKEND="cuda12"
        else
            BACKEND="cpu"
        fi
        ;;
    cuda12)
        [[ "$(uname -s)" == "Linux" ]] || die "JAX CUDA wheels require Linux; use the cpu backend on this OS."
        command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found; install an NVIDIA driver or use cpu."
        ;;
    cpu) ;;
    *) die "backend must be one of: auto, cuda12, cpu" ;;
esac

case "${INSTALL_PROFILE}" in
    train|metrics) ;;
    *) die "install profile must be one of: train, metrics" ;;
esac

printf 'Conda environment: %s\n' "${ENV_NAME}"
printf 'JAX backend:      %s\n' "${BACKEND}"

if conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
    INSTALLED_PYTHON="$(conda run -n "${ENV_NAME}" python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${INSTALLED_PYTHON}" == "${PYTHON_VERSION}" ]] || die \
        "Existing environment ${ENV_NAME} uses Python ${INSTALLED_PYTHON}; expected ${PYTHON_VERSION}. Choose another environment name."
    printf 'Reusing existing conda environment.\n'
else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

if [[ "${BACKEND}" == "cuda12" ]]; then
    JAX_PACKAGE='jax[cuda12]==0.6.2'
else
    JAX_PACKAGE='jax==0.6.2'
fi

# Conda provides only Python and pip. All runtime packages are installed by pip.
conda run -n "${ENV_NAME}" python -m pip install \
    "${JAX_PACKAGE}" \
    'diffrax==0.7.0' \
    'equinox==0.13.8' \
    'jaxtyping==0.3.7' \
    'matplotlib==3.10.9' \
    'optax==0.2.8' \
    'wandb==0.27.0'

if [[ "${INSTALL_PROFILE}" == "metrics" ]]; then
    if [[ "${BACKEND}" == "cuda12" ]]; then
        TORCH_INDEX='https://download.pytorch.org/whl/cu126'
        conda run -n "${ENV_NAME}" python -m pip install \
            --index-url "${TORCH_INDEX}" torch torchvision
    elif [[ "$(uname -s)" == "Linux" ]]; then
        TORCH_INDEX='https://download.pytorch.org/whl/cpu'
        conda run -n "${ENV_NAME}" python -m pip install \
            --index-url "${TORCH_INDEX}" torch torchvision
    else
        conda run -n "${ENV_NAME}" python -m pip install torch torchvision
    fi
fi

printf 'Running import and tiny-model smoke test...\n'
conda run -n "${ENV_NAME}" python - "${MODEL_PATH}" "${BACKEND}" <<'PY'
import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp

model_path = pathlib.Path(sys.argv[1])
backend = sys.argv[2]
spec = importlib.util.spec_from_file_location("ssfm_cifar", model_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

key = jax.random.PRNGKey(0)
model = module.build_model(
    key,
    resolution=8,
    base_channels=8,
    channel_mult=(1,),
    num_groups=1,
    dropout_rate=0.0,
    attn_resolutions=(),
    head_dim=8,
    num_res_blocks=1,
    diff_base_channels=8,
    diff_channel_mult=(1,),
    diff_attn_resolutions=(),
)
x = jnp.zeros((3, 8, 8))
w = jnp.zeros_like(x)
out = model(x, jnp.asarray(1.0), jnp.asarray(0.5), w, w, w, key=key)
assert out.shape == x.shape, (out.shape, x.shape)

devices = jax.devices()
if backend == "cuda12" and not any(device.platform == "gpu" for device in devices):
    raise RuntimeError(f"CUDA backend requested, but JAX found no GPU: {devices}")

print(f"JAX {jax.__version__}; devices={devices}")
print(f"Smoke test passed; output shape={out.shape}")
PY

printf '\nEnvironment setup completed.\n'
printf 'Activate: conda activate %s\n' "${ENV_NAME}"
printf 'Train:    python %s\n' "${MODEL_PATH}"
printf 'Offline W&B: WANDB_MODE=offline python %s\n' "${MODEL_PATH}"
