# behavior1k-v1 수정 요약

## 2026-05-06 System 2 stage tracking 추가

이번 작업에서는 `behavior1k-v1`의 기존 목적이었던 **task embedding + flow matching + correlated noise** 구조를 유지하면서,
그 위에 1등팀 방식에 가까운 **System 2 stage tracking** 실험 경로를 추가했다.

쉽게 말하면, 기존 v1은 모델에게 "몇 번 task인지"만 알려주는 구조였다.
이번에 추가한 stage tracking 버전은 여기에 "지금 그 task의 몇 번째 단계인지"까지 같이 알려준다.
예를 들어 방 정리 task라면 단순히 "방 정리"만 주는 것이 아니라,
"지금은 물건 찾기 단계인지, 옮기기 단계인지, 정리 마무리 단계인지"를 모델 입력에 같이 넣을 수 있게 만든 것이다.

추가된 핵심 config:

```bash
pi_behavior_b1k_a100_week_stage
```

이 config의 의미:

- `pi_behavior_b1k_a100_week`의 1주일 이내 A100 실험용 설정을 기반으로 한다.
- v1의 핵심 차이점인 `correlated noise`는 계속 켜져 있다.
- `subtask_loss_weight=0.1`로 stage 예측 보조 loss를 켰다.
- `use_stage_conditioning=True`로 데이터 transform에서 stage 정보를 모델 입력에 붙인다.
- 모델 입력 token은 기존 `[task_id]`에서 `[task_id, stage_id]` 형태가 된다.

수정된 주요 파일:

- `src/b1k/transforms.py`
  - dataset metadata에서 episode 진행률을 읽어 현재 stage를 계산한다.
  - global task id를 local task id로 바꾼 뒤 stage 개수를 찾도록 수정했다.
  - stage tracking을 켜면 `tokenized_prompt`에 task id와 stage id를 같이 넣는다.

- `src/b1k/training/config.py`
  - `use_stage_conditioning` 옵션을 추가했다.
  - `pi_behavior_b1k_a100_week_stage` config를 추가했다.

- `src/b1k/training/data_loader.py`
  - stage 계산 transform이 실제 dataset metadata를 볼 수 있도록 dataset 객체를 연결한다.

- `src/b1k/models/pi_behavior.py`
  - 학습 중 stage 예측 loss와 accuracy를 기록하도록 했다.
  - 평가/서빙에서 필요한 `sample_actions()`를 추가했다.
  - action sampling은 flow matching 경로를 사용하고, stage tracking을 켠 경우 stage logit도 함께 반환한다.

- `src/b1k/shared/eval_b1k_wrapper.py`
  - 서빙/평가 중 모델이 예측한 stage를 바로 믿지 않고, 최근 예측을 모아 다수결로 현재 stage를 갱신한다.
  - stage tracking을 켜면 wrapper가 `[local_task_id, current_stage]`를 모델에 넣는다.

- `scripts/serve_b1k.py`
  - 기본 서빙 config 예시를 `pi_behavior_b1k_a100_week_stage`로 바꿨다.
  - `--no-use-stage-tracking` 옵션으로 stage tracking을 끌 수 있게 했다.

실행 예시:

```bash
uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_a100_week_stage --correlation
uv run scripts/train.py pi_behavior_b1k_a100_week_stage --overwrite
```

중간에 끊긴 학습을 이어서 할 때:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week_stage --resume
```

서빙 예시:

```bash
uv run scripts/serve_b1k.py \
  --policy.dir ./checkpoints/pi_behavior_b1k_a100_week_stage/a100_week_stage_10k/10000 \
  --policy.config pi_behavior_b1k_a100_week_stage \
  --task-id 0
```

주의할 점:

- 이 버전은 1등팀 전체 코드를 그대로 복사한 것이 아니라, v1의 correlated noise 실험 구조 위에 stage tracking을 추가한 버전이다.
- 1등팀의 모든 inference trick, 전체 task group fine-tuning, full-scale H200 학습 조건까지 재현한 것은 아니다.
- 실제 A100 학습 시간과 성능은 dataset 위치, batch size, GPU 종류, checkpoint 저장 주기, OmniGibson 평가 환경에 따라 달라진다.
- 이번 검증은 코드 문법/구조 검증 중심이며, 실제 A100 장시간 학습과 OmniGibson rollout은 로컬에서 실행하지 않았다.

이 저장소는 `SUN-STAR-HASH/behavior1k`를 직접 수정하지 않고, 별도 새 레포로 공개하기 위한 `v1` 버전이다.

핵심 방향은 다음과 같다.

- 기존 task embedding + flow matching baseline 유지
- `IliaLarchenko/behavior-1k-solution`의 correlated noise 아이디어만 우선 반영
- README, 실행 예시, 메타데이터를 새 레포 기준으로 정리

## 새 대표 config

`pi_behavior_b1k_v1`

구성:

- correlated noise ON
- `correlation_beta=0.5`
- FAST auxiliary OFF
- KV transform OFF
- subtask/stage loss OFF

즉, `v1`은 "1등팀 전체 기능 이식본"이 아니라 "baseline + correlated noise" 버전이다.

## 실험 흐름

baseline 비교:

```bash
uv run scripts/train.py pi_behavior_b1k_baseline --overwrite
```

v1 correlated noise:

```bash
uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_v1 --correlation
uv run scripts/train.py pi_behavior_b1k_v1 --overwrite
```

짧은 A100 확인용:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week --overwrite
```

## 문서 정리 범위

- `README.md` 전면 교체
- `pyproject.toml` 이름/설명 수정
- `scripts/serve_b1k.py` 예시 config 이름 정리
- `src/b1k/training/config.py`에 `pi_behavior_b1k_v1` 추가
