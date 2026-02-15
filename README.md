# Snake RL

Snake RL playground with a Rust core engine and Python training/eval/watch tooling.

## Highlights
- Rust-backed engine (deterministic RNG, fast vector stepping)
- Stable-Baselines3 PPO and RecurrentPPO support
- Pixel and categorical observations (`world` and `head` views)
- Resume and refine workflows for existing runs
- Pygame watch mode with optional CNN visualization

## Installation

Requirements:
- Python 3.10+
- Rust toolchain (`cargo`, `rustc`)

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Optional extras:
- `pip install -e ".[viz]"` for OpenCV CNN visualization backend.

## Quick Start

Train using the provided preset:
```powershell
snake-train -cfg configs/train_head_pixel_ppo.yaml
```

Train with common overrides:
```powershell
snake-train -cfg configs/train_head_pixel_ppo.yaml --override run.seed=123 --override run.num_envs=8 --override run.total_timesteps=5_000_000
```

Resume a finished run for more steps:
```powershell
snake-train --resume runs\head_pixel_ppo_classic_001 --resume-add-steps 2_000_000
```

Refine from an existing checkpoint into a new run:
```powershell
snake-train --refine runs\head_pixel_ppo_classic_001 --refine-from best_win --refine-steps 5_000_000 --refine-lr 1e-5 --refine-ent 0.0
```

Evaluate:
```powershell
snake-eval --run runs\head_pixel_ppo_classic_001 --which auto --episodes 100 --deterministic
```

Watch:
```powershell
snake-watch --run runs\head_pixel_ppo_classic_001
snake-watch --run runs\head_pixel_ppo_classic_001 --reload 30
```

Watch bundled example:
```powershell
snake-watch --run examples\head_pixel_ppo_classic_005
```

Play manually:
```powershell
snake-play
```

Benchmark:
```powershell
snake-bench --mode engine
snake-bench --mode env --config train_head_pixel_ppo.yaml
```

## Configs

Config root is `configs/`.

Primary tracked training preset:
- `configs/train_head_pixel_ppo.yaml`

Base groups:
- `configs/run/default.yaml`
- `configs/board/default.yaml`
- `configs/reward/default.yaml`
- `configs/env/default.yaml`
- `configs/algo/default.yaml`
- `configs/feature_extractor/snake_pixel_cnn_head.yaml`
- `configs/metrics/default.yaml`
- `configs/logging/default.yaml`

Important note:
- `configs/config.yaml` is a base composition and does not pin a feature extractor group by default.
- Use a train preset (recommended) or pass explicit overrides when using `configs/config.yaml`.

Local experiments:
- `configs/local/` is intended for local experimentation and is gitignored.

## Checkpoints

`snake-watch` and `snake-eval` accept `--which`:
- `auto`, `latest`, `best_reward`, `best_score`, `best_win`, `best`, `final`

Current `auto` preference:
1. `latest`
2. `best_reward`
3. `best_score`
4. `best_win`
5. `best` (legacy fallback)

## Run Layout

```text
runs/<run_id>/
  config_snapshot.yaml
  config_summary.yaml
  config_hydra.yaml
  config_validated.yaml
  checkpoints/
    latest.zip
    best_reward.zip
    best_score.zip
    best_win.zip
    final.zip
    eval_history.jsonl
    state.json
  tb/
  eval_final.json
  status.txt
```

Legacy run loading via `experiments/` is still supported.

## Dev Checks

```powershell
python -m ruff check src/snake_rl
python -m pyright
```

```powershell
pre-commit install
pre-commit run --all-files
```

```powershell
cargo clippy --all-targets --all-features -- -D warnings
```

## What Is Still Missing

- A full config reference with all keys and defaults (especially feature extractor and reward shaping).
- A troubleshooting section (CUDA/PyTorch mismatch, pygame warnings, checkpoint loading pitfalls).
- CI status badges and documented required checks before merge.
- A reproducible benchmark table (hardware, command, expected steps/s).

## License

MIT. See `LICENSE`.
