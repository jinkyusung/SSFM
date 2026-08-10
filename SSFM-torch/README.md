# SSFM Torch

PyTorch로 독립 구현한 CIFAR-10 Strong Stochastic Flow Map입니다. 모델, VP diffusion,
`W/H/K` Levy area 생성과 Chen 결합, uncertainty score/distillation loss, EMA,
샘플링, 체크포인트, FID 평가를 모두 이 디렉터리 안에서 제공합니다.

이 구현의 실행 의존성에는 JAX, Equinox, Diffrax, Optax가 없습니다. 체크포인트도
Torch 네이티브 `state.pt` 형식이므로 기존 `.eqx` 파일을 실행 중에 읽지 않습니다.

## 설치

```bash
cd SSFM-torch
./setup_env.sh auto
conda activate ssfm-torch
```

직접 설치하려면 다음을 사용합니다.

```bash
python -m pip install -e '.[test]'
pytest
```

## 학습

```bash
python train.py
```

기본값으로 CIFAR-10은 저장소 최상단 `data/`를 JAX 구현과 공유합니다.
체크포인트, 모델, W&B 파일과 Slurm 로그는 각각 `SSFM-torch/checkpoints/`,
`SSFM-torch/models/`, `SSFM-torch/wandb/`, `SSFM-torch/slurm/`에 저장됩니다.

두 장 이상의 NVIDIA GPU에서는 `torchrun`으로 DDP 학습을 실행할 수 있습니다.
`--batch-size`는 모든 프로세스를 합친 전역 배치 크기입니다.

```bash
torchrun --standalone --nproc-per-node=gpu train.py
```

Slurm용 `run.sh`도 기본적으로 할당된 GPU 전체에 `torchrun`을 사용합니다.

빠른 실행 확인 예시는 다음과 같습니다.

```bash
WANDB_MODE=disabled python train.py --steps 1 --batch-size 2 --device cpu
```

기본 하이퍼파라미터와 네트워크 구조는 원 구현과 같습니다. 입력 데이터는
CIFAR-10의 각 픽셀 위치별 평균과 표준편차로 정규화됩니다. 체크포인트는 가장
최근의 `checkpoints/<step>/state.pt`에서 자동 재개됩니다.

## 평가

```bash
python eval_fid.py checkpoints/400k
python bm_consistency.py checkpoints/400k --output bm_consistency.png
```

모델 API는 Torch 표준 `NCHW` 배치를 지원하며, 한 이미지의 `CHW` 입력도
지원합니다.

```python
import torch
from ssfm_torch import build_model

model = build_model().eval()
y = torch.randn(4, 3, 32, 32)
W, H, K = (torch.randn_like(y) for _ in range(3))
output = model(y, 1.0, 0.5, W, H, K)
```
