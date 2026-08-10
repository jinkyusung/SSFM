# SSFM JAX

JAX, Equinox, Diffrax로 구현한 CIFAR-10 Strong Stochastic Flow Map 기준
구현입니다. CIFAR-10 데이터는 두 구현이 공유하는 저장소 최상단 `data/`를
사용합니다. 체크포인트, 모델, 샘플 결과, W&B와 Slurm 로그는 현재 작업
디렉터리와 관계없이 `SSFM-jax/` 아래에 저장됩니다.

## 환경 설정

저장소 루트에서:

```bash
./SSFM-jax/setup_ssfm_cifar_env.sh auto
conda activate ssfm-cifar
```

`auto` 대신 `cuda12` 또는 `cpu`를 지정할 수 있습니다. FID 평가용 PyTorch
패키지까지 설치하려면 다음을 사용합니다.

```bash
./SSFM-jax/setup_ssfm_cifar_env.sh auto ssfm-cifar metrics
```

## 학습

어느 작업 디렉터리에서든 다음처럼 실행할 수 있습니다.

```bash
python /path/to/SSFM/SSFM-jax/ssfm_cifar.py
```

저장소 루트에서는 다음 명령으로 충분합니다.

```bash
python SSFM-jax/ssfm_cifar.py
sbatch SSFM-jax/run.sh
```

다른 환경 이름을 사용했다면 Slurm 제출 시 동일한 이름을 전달합니다.

```bash
sbatch --export=ALL,SSFM_CONDA_ENV=my-ssfm-env SSFM-jax/run.sh
```

CIFAR-10은 저장소 최상단 `data/`, 체크포인트는 `SSFM-jax/checkpoints/`,
모델은 `SSFM-jax/models/`, W&B 파일은 `SSFM-jax/wandb/`, Slurm 로그는
`SSFM-jax/slurm/` 아래에 저장됩니다. `WANDB_MODE=offline`으로 오프라인
학습을 실행할 수 있습니다.

## 평가

```bash
python SSFM-jax/eval_fid.py
python SSFM-jax/bm_consistency.py
```

Slurm에서 평가 파일을 선택할 수도 있습니다.

```bash
sbatch SSFM-jax/run.sh eval_fid.py
sbatch SSFM-jax/run.sh bm_consistency.py
```
