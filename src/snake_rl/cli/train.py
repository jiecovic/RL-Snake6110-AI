# src/snake_rl/cli/train.py
from __future__ import annotations

import argparse
from pathlib import Path

from snake_rl.config.io import apply_overrides, load_config
from snake_rl.training.train_loop import train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on Snake (headless).")
    p.add_argument("--config", type=str, default="configs/example.yaml", help="Path to YAML config.")
    p.add_argument("--seed", type=int, default=None, help="Override run.seed.")
    p.add_argument("--num-envs", type=int, default=None, help="Override run.num_envs.")
    p.add_argument("--total-timesteps", type=int, default=None, help="Override run.total_timesteps.")
    p.add_argument("--checkpoint-freq", type=int, default=None, help="Override run.checkpoint_freq.")
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint .zip path OR a run id OR 'latest:<run_name>'. Overrides run.resume_checkpoint.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = load_config(Path(args.config))
    cfg = apply_overrides(
        cfg,
        seed=args.seed,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        checkpoint_freq=args.checkpoint_freq,
        resume_checkpoint=args.resume,
    )

    # resume is already applied into cfg.run.resume_checkpoint
    train(cfg=cfg)


if __name__ == "__main__":
    main()
