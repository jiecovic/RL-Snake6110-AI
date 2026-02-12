# src/snake_rl/cli/train.py
from __future__ import annotations

import argparse
from pathlib import Path

from snake_rl.app.train_app import run_from_config_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Snake PPO agent.")
    p.add_argument(
        "-c",
        "-cfg",
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config path (file or configs/ entry).",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable). Only used when --config is under configs/.",
    )
    p.add_argument("--no-rich", action="store_true", help="Disable Rich logging.")
    p.add_argument("--log-level", type=str, default=None, help="Override logging level.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_from_config_path(
        config_path=Path(args.config),
        overrides=list(args.override),
        no_rich=bool(args.no_rich),
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
