# src/snake_rl/cli/train.py
from __future__ import annotations

import argparse
from pathlib import Path

from snake_rl.app.train_app import run_from_config_path
from snake_rl.utils.runs.paths import repo_root, runs_root
from snake_rl.utils.runs.resume import infer_config_path_from_resume


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Snake PPO agent.")
    p.add_argument(
        "-c",
        "-cfg",
        "--config",
        type=str,
        default=None,
        help="Config path (file or configs/ entry).",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable). Only used when --config is under configs/.",
    )
    p.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from checkpoint, run id/path, or latest:<prefix>. "
            "If --config is omitted, tries to load runs/<run>/config_snapshot.yaml."
        ),
    )
    p.add_argument("--no-rich", action="store_true", help="Disable Rich logging.")
    p.add_argument("--log-level", type=str, default=None, help="Override logging level.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config) if args.config else None
    if config_path is None:
        if args.resume:
            config_path = infer_config_path_from_resume(
                str(args.resume),
                runs_root=runs_root(),
                legacy_root=repo_root() / "experiments",
            )
        else:
            config_path = Path("configs/config.yaml")
    run_from_config_path(
        config_path=config_path,
        overrides=list(args.override),
        no_rich=bool(args.no_rich),
        log_level=args.log_level,
        resume_override=args.resume,
    )


if __name__ == "__main__":
    main()
