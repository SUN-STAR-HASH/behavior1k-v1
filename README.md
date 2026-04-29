# behavior1k-v1

`behavior1k-v1` is a separate repository version of [`SUN-STAR-HASH/behavior1k`](https://github.com/SUN-STAR-HASH/behavior1k) for a focused BEHAVIOR-1K baseline experiment.

This version keeps the baseline direction centered on:

- task embeddings instead of language prompts
- flow matching action modeling
- winner-style correlated noise added from [`IliaLarchenko/behavior-1k-solution`](https://github.com/IliaLarchenko/behavior-1k-solution)

The original `SUN-STAR-HASH/behavior1k` repository is not modified. This repo is intended to be published separately.

## Scope Of v1

The goal of `v1` is not to copy every feature from the 1st-place solution.

Instead, it isolates one main change on top of the baseline:

- baseline: task embedding + flow matching
- v1: baseline + correlated noise

That makes it easier to compare:

- `pi_behavior_b1k_baseline`
- `pi_behavior_b1k_v1`

## Main Configs

This repository contains several smoke/debug presets from previous experiments, but the important ones are:

- `pi_behavior_b1k_baseline`
- `pi_behavior_b1k_v1`
- `pi_behavior_b1k_a100_week`

Use `pi_behavior_b1k_v1` if you want the new correlated-noise version.

## What Changes In `pi_behavior_b1k_v1`

`pi_behavior_b1k_v1` keeps the lightweight baseline shape and turns on only correlated noise:

- `use_correlated_noise=True`
- `correlation_beta=0.5`
- `use_fast_auxiliary=False`
- `use_kv_transform=False`
- `subtask_loss_weight=0.0`
- `freeze_vision_backbone=True`

So this is a clean "baseline + correlated noise" experiment branch.

## Repository Layout

```text
src/b1k/
  models/          PiBehavior model and config
  training/        training configs, dataloader, checkpoint logic
  policies/        policy loading and eval-time wrappers
  shared/          normalization and evaluation helpers
scripts/
  compute_norm_stats.py
  train_fast_tokenizer.py
  train.py
  serve_b1k.py
BEHAVIOR-1K/       official simulator/eval submodule
openpi/            OpenPI submodule
```

## Installation

Clone with submodules:

```bash
git clone --recurse-submodules <NEW-REPO-URL>
cd behavior1k-v1
```

Install:

```bash
bash setup_remote.sh
```

If submodules are empty:

```bash
git submodule update --init --recursive
```

Recommended environment:

- Linux
- Python 3.11
- CUDA 12.x
- NVIDIA GPU

## Dataset Preparation

The default configs use the resized RGB dataset:

- `IliaLarchenko/behavior_224_rgb`

Example download:

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

Expected layout:

```text
<data_root>/data/task-*/episode_*.parquet
```

Update data and output paths in `src/b1k/training/config.py` if needed.

## Preprocessing

For `v1`, compute normalization statistics with correlation enabled:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi_behavior_b1k_v1 \
  --correlation
```

This step is important because correlated-noise sampling depends on correlation statistics saved in the assets directory.

FAST tokenizer training is not required for `pi_behavior_b1k_v1`, because FAST is intentionally disabled there.

## Training

### Baseline

```bash
uv run scripts/train.py pi_behavior_b1k_baseline --overwrite
```

### v1: baseline + correlated noise

```bash
uv run scripts/train.py pi_behavior_b1k_v1 --overwrite
```

Resume:

```bash
uv run scripts/train.py pi_behavior_b1k_v1 --resume
```

Default `v1` settings:

- `num_train_steps=30000`
- `batch_size=16`
- `num_flow_samples=1`
- correlated noise ON
- FAST OFF
- KV transform OFF
- subtask loss OFF

### Short A100 Week Run

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week --overwrite
```

This config is still useful when you want a shorter practical run rather than the longer `v1` experiment.

### Optional W&B

```bash
uv run wandb login
```

Disable W&B logging:

```bash
uv run scripts/train.py pi_behavior_b1k_v1 --wandb_enabled=false
```

## Evaluation

Start the websocket policy server:

```bash
uv run scripts/serve_b1k.py \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_v1 \
  --policy.dir /path/to/checkpoint
```

Multi-checkpoint mode:

```bash
uv run scripts/serve_b1k.py \
  --task-checkpoint-mapping task_checkpoint_mapping.json \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_v1 \
  --policy.dir /path/to/initial/checkpoint
```

Then run evaluation from another terminal:

```bash
python BEHAVIOR-1K/omnigibson/learning/eval.py \
  log_path=./eval_logs \
  policy=websocket \
  model.host=localhost \
  model.port=8000 \
  task.name=make_microwave_popcorn \
  eval_instance_ids="[0,1,2,3]"
```

## References

- Original fork source: [SUN-STAR-HASH/behavior1k](https://github.com/SUN-STAR-HASH/behavior1k)
- 1st-place reference: [IliaLarchenko/behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution)
- BEHAVIOR-1K: [StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
- OpenPI: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)

## License

Apache-2.0
