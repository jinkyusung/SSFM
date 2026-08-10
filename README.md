# SSFM for CIFAR-10

이 저장소는 동일한 Strong Stochastic Flow Map을 두 개의 독립 구현으로
제공합니다.

- [SSFM-jax](SSFM-jax/README.md): JAX, Equinox, Diffrax 기준 구현
- [SSFM-torch](SSFM-torch/README.md): JAX 런타임 의존성이 없는 PyTorch 구현

각 디렉터리의 README에 환경 설정, 학습, 평가 방법이 정리되어 있습니다.

CIFAR-10 데이터는 최상단 `data/`에서 두 구현이 공유합니다. 체크포인트,
모델, W&B와 Slurm 로그 같은 실행 결과는 각 구현 디렉터리 내부에 분리됩니다.
