# behavior1k-v1

`behavior1k-v1`는 [`SUN-STAR-HASH/behavior1k`](https://github.com/SUN-STAR-HASH/behavior1k)를 직접 수정하지 않고, 별도 저장소로 분리한 BEHAVIOR-1K 실험용 버전입니다.

이 저장소의 핵심 방향은 다음과 같습니다.

- 자연어 프롬프트 대신 task embedding 사용
- action modeling은 flow matching 기반 유지
- [`IliaLarchenko/behavior-1k-solution`](https://github.com/IliaLarchenko/behavior-1k-solution)의 아이디어를 참고해 correlated noise를 추가

즉, 이 저장소는 "1등팀 전체 기능 이식본"이 아니라, 기존 baseline 위에 correlated noise만 우선 얹어서 비교하기 쉽게 만든 버전입니다.

## v1의 목적

`v1`의 목표는 기능을 많이 넣는 것이 아니라, baseline과 비교 가능한 한 가지 실험축을 분명하게 만드는 것입니다.

- baseline: task embedding + flow matching
- v1: baseline + correlated noise

이렇게 나누면 아래 두 설정을 직접 비교하기 쉽습니다.

- `pi_behavior_b1k_baseline`
- `pi_behavior_b1k_v1`

## 주요 config

이 저장소에는 smoke/debug용 설정도 같이 들어 있지만, 실제로 중요한 설정은 아래 3개입니다.

- `pi_behavior_b1k_baseline`
- `pi_behavior_b1k_v1`
- `pi_behavior_b1k_a100_week`

새 correlated-noise 버전을 쓰고 싶다면 `pi_behavior_b1k_v1`를 사용하면 됩니다.

## `pi_behavior_b1k_v1`에서 달라진 점

`pi_behavior_b1k_v1`는 baseline 구조를 최대한 유지한 상태에서 correlated noise만 켭니다.

- `use_correlated_noise=True`
- `correlation_beta=0.5`
- `use_fast_auxiliary=False`
- `use_kv_transform=False`
- `subtask_loss_weight=0.0`
- `freeze_vision_backbone=True`

따라서 `v1`은 "baseline + correlated noise" 실험 브랜치라고 보면 됩니다.

## 저장소 구조

```text
src/b1k/
  models/          PiBehavior 모델 및 설정
  training/        학습 config, dataloader, checkpoint 로직
  policies/        policy 로딩 및 평가용 wrapper
  shared/          normalization, eval helper
scripts/
  compute_norm_stats.py
  train_fast_tokenizer.py
  train.py
  serve_b1k.py
BEHAVIOR-1K/       공식 simulator/eval 서브모듈
openpi/            OpenPI 서브모듈
```

## 설치

서브모듈까지 포함해서 clone:

```bash
git clone --recurse-submodules https://github.com/SUN-STAR-HASH/behavior1k-v1.git
cd behavior1k-v1
```

환경 설치:

```bash
bash setup_remote.sh
```

서브모듈이 비어 있다면:

```bash
git submodule update --init --recursive
```

권장 환경:

- Linux
- Python 3.11
- CUDA 12.x
- NVIDIA GPU

## 데이터셋 준비

기본 설정은 resized RGB dataset을 사용합니다.

- `IliaLarchenko/behavior_224_rgb`

예시 다운로드:

```bash
uv run huggingface-cli login

uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IliaLarchenko/behavior_224_rgb",
    repo_type="dataset",
    local_dir="./data/behavior_224_rgb",
    local_dir_use_symlinks=False,
)
PY
```

예상 데이터 구조:

```text
<data_root>/data/task-*/episode_*.parquet
```

필요하면 `src/b1k/training/config.py`에서 데이터 경로와 출력 경로를 수정하면 됩니다.

## 전처리

`v1`에서는 correlated noise를 쓰기 때문에 correlation 통계를 포함한 normalization statistics가 필요합니다.

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi_behavior_b1k_v1 \
  --correlation
```

이 단계에서 생성된 correlation 정보가 correlated-noise sampling에 사용됩니다.

`pi_behavior_b1k_v1`에서는 FAST를 일부러 끄기 때문에 FAST tokenizer 학습은 필요하지 않습니다.

## 학습

### 1. baseline

```bash
uv run scripts/train.py pi_behavior_b1k_baseline --overwrite
```

### 2. v1: baseline + correlated noise

```bash
uv run scripts/train.py pi_behavior_b1k_v1 --overwrite
```

이어서 학습:

```bash
uv run scripts/train.py pi_behavior_b1k_v1 --resume
```

기본 `v1` 설정:

- `num_train_steps=30000`
- `batch_size=16`
- `num_flow_samples=1`
- correlated noise ON
- FAST OFF
- KV transform OFF
- subtask loss OFF

### 3. 짧은 A100 실험용 설정

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week --overwrite
```

이 설정은 긴 `v1` 실험보다 빠르게 확인하고 싶을 때 유용합니다.
이 레포에서는 이 A100 week 설정도 `v1` 방향에 맞춰 correlated noise를 켠 상태로 둡니다.

### W&B 사용

```bash
uv run wandb login
```

W&B를 끄고 싶다면:

```bash
uv run scripts/train.py pi_behavior_b1k_v1 --wandb_enabled=false
```

## 평가

websocket policy server 실행:

```bash
uv run scripts/serve_b1k.py \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_v1 \
  --policy.dir /path/to/checkpoint
```

multi-checkpoint 모드:

```bash
uv run scripts/serve_b1k.py \
  --task-checkpoint-mapping task_checkpoint_mapping.json \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_v1 \
  --policy.dir /path/to/initial/checkpoint
```

다른 터미널에서 evaluation 실행:

```bash
python BEHAVIOR-1K/omnigibson/learning/eval.py \
  log_path=./eval_logs \
  policy=websocket \
  model.host=localhost \
  model.port=8000 \
  task.name=make_microwave_popcorn \
  eval_instance_ids="[0,1,2,3]"
```

## 참고

- 원본 fork: [SUN-STAR-HASH/behavior1k](https://github.com/SUN-STAR-HASH/behavior1k)
- 1등팀 참고 구현: [IliaLarchenko/behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution)
- BEHAVIOR-1K: [StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
- OpenPI: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

## 라이선스

Apache-2.0
