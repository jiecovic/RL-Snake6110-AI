# src/snake_rl/cli/train.py
from __future__ import annotations

import argparse
from pathlib import Path

from snake_rl.app.train_app import run_from_config_path, run_refine
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
    p.add_argument(
        "--refine",
        type=str,
        default=None,
        help="Refine from an existing run or checkpoint (creates a new run dir).",
    )
    p.add_argument(
        "--refine-from",
        type=str,
        default="latest",
        help="Refine source: latest|final|best_reward|best_score|best_win|best",
    )
    p.add_argument(
        "--refine-steps",
        type=int,
        default=None,
        help="Number of steps to train in the refined run (required with --refine).",
    )
    p.add_argument("--refine-lr", type=float, default=None, help="Override learning rate.")
    p.add_argument("--refine-ent", type=float, default=None, help="Override ent_coef.")
    p.add_argument(
        "--refine-suffix",
        type=str,
        default="refined",
        help="Suffix for refined run name (default: refined).",
    )
    p.add_argument(
        "--refine-name",
        type=str,
        default=None,
        help="Explicit refined run base name (overrides auto naming).",
    )
    p.add_argument(
        "--resume-total-steps",
        type=int,
        default=None,
        help="Override run.total_timesteps when resuming (absolute).",
    )
    p.add_argument(
        "--resume-add-steps",
        type=int,
        default=None,
        help="Add N steps to run.total_timesteps when resuming (relative).",
    )
    p.add_argument("--no-rich", action="store_true", help="Disable Rich logging.")
    p.add_argument("--log-level", type=str, default=None, help="Override logging level.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.refine and args.resume:
        raise SystemExit("Use only one of --resume or --refine.")
    if args.refine and args.refine_steps is None:
        raise SystemExit("--refine-steps is required with --refine.")
    config_path = Path(args.config) if args.config else None
    if args.refine:
        run_refine(
            refine=str(args.refine),
            refine_from=str(args.refine_from) if args.refine_from else None,
            refine_steps=int(args.refine_steps),
            refine_lr=args.refine_lr,
            refine_ent=args.refine_ent,
            refine_suffix=str(args.refine_suffix) if args.refine_suffix else None,
            refine_name=args.refine_name,
            config_path=config_path,
            overrides=list(args.override),
            no_rich=bool(args.no_rich),
            log_level=args.log_level,
        )
        return

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
        resume_total_steps=args.resume_total_steps,
        resume_add_steps=args.resume_add_steps,
    )


if __name__ == "__main__":
    main()
