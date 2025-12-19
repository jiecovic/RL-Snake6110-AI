# src/snake_rl/cli/train.py
from __future__ import annotations

import argparse
from pathlib import Path

from snake_rl.config.io import apply_overrides, load_config
from snake_rl.training.train_loop import train


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on Snake (headless).")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config.")
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

    # Logging cosmetics (match watch.py ergonomics)
    p.add_argument("--no-rich", action="store_true", help="Disable Rich logging (fallback to plain logging).")
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")

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

    train(cfg=cfg, use_rich=not bool(args.no_rich), log_level=str(args.log_level))


if __name__ == "__main__":
    main()
