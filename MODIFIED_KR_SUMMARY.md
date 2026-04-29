# behavior1k-v1 수정 요약

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
