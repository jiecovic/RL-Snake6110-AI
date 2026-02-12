# Snake RL - PPO Agent for Classic Snake

This project implements the classic Snake game and provides an RL playground for
training agents (currently PPO via Stable-Baselines3). The core game mechanics
live in a Rust engine for speed and determinism, with Python wrappers for
training, evaluation, and visualization.

## Highlights
- Rust-backed Snake engine (fast, deterministic RNG)
- PPO agents with Stable-Baselines3 + Gymnasium
- Multiple observation modes (pixels, head/world views, categorical grids)
- Reproducible runs via config snapshots
- TensorBoard logging and live watch mode

## Installation

Requirements:
- Python 3.10+
- Rust toolchain (cargo/rustc) for the native core

Create a single virtual environment and install editable + dev tools:

Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Linux/macOS:
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

PyTorch install guidance: https://pytorch.org/get-started/locally/

## Quick Start

Train a PPO agent:
```
snake-train --config-name main_head_pixel_ppo
```

Common overrides:
```
snake-train --config-name main_head_pixel_ppo run.seed=123 run.num_envs=8 run.total_timesteps=5_000_000
```

Use the Rust vectorized engine for higher throughput:
```
snake-train --config-name main_head_pixel_ppo env.engine=rust
```

Evaluate a trained run:
```
snake-eval --run snake_ppo_001
```

Watch a trained run (live reload optional):
```
snake-watch --run snake_ppo_001
snake-watch --run snake_ppo_001 --reload 30
```

Play as a human:
```
snake-play
snake-play --width 30 --height 20 --fps 15
```

## Configuration

Hydra config groups live under `configs/`:
- `board/` (grid sizes)
- `env/` (env id + params)
- `reward/` (reward shaping)
- `feature_extractor/` (feature extractor params)
- `train/` (algorithm + eval scheduling)
- `run/` (seed, num_envs, total_timesteps, checkpoints)

Main configs live at the top level: `configs/main_*.yaml`.

Notes:
- `train.algo.params.policy_kwargs.net_arch` controls the policy MLP layout.
- `train.eval` applies to both periodic eval during training and the final eval pass.

## Run Structure

```
runs/
  snake_ppo_001/
    config_snapshot.yaml
    config_hydra.yaml
    config_validated.yaml
    checkpoints/
      latest.zip
      best.zip
      final.zip
    tb/
    eval_final.json
    status.txt
```

Legacy runs under `experiments/` are still supported for loading.

## Dev Tools

One-command checks:
```
snake-check
```

Manual checks:
```
python -m ruff check src/snake_rl
python -m pyright
```

Pre-commit:
```
pre-commit install
pre-commit run --all-files
```

Rust lint (clippy):
```
cargo clippy --all-targets --all-features -- -D warnings
```

## License

MIT License - see LICENSE
