# src/snake_rl/cli/eval.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO

from snake_rl.config.frozen import get_run_num_envs, get_run_seed, resolve_cli_config
from snake_rl.training.eval_utils import evaluate_model
from snake_rl.training.reporting import log_ppo_params
from snake_rl.utils.checkpoints import pick_checkpoint
from snake_rl.utils.logging import setup_logger
from snake_rl.utils.models import load_ppo
from snake_rl.utils.paths import repo_root, resolve_run_dir

try:
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
except Exception:  # pragma: no cover
    Progress = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _EvalPaths:
    repo_root: Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Snake PPO checkpoint/model.")
    p.add_argument("--run", type=str, required=True, help="Run id / run folder name under experiments/")
    p.add_argument("--which", type=str, default="latest", choices=["auto", "latest", "best", "final"])
    p.add_argument("--episodes", type=int, default=20)

    # None => default from config_frozen.yaml (run.num_envs), clamped to <= episodes
    p.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Parallel eval envs. Default: run.num_envs from config_frozen.yaml.",
    )

    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Default: cfg.run.seed + 12345 (from config_frozen.yaml).",
    )

    # Config resolution: YAML only
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path. If omitted, uses experiments/<run>/config_frozen.yaml.",
    )

    # Output controls
    p.add_argument("--json", action="store_true", help="Print full JSON metrics to stdout.")
    p.add_argument("--out", type=str, default=None, help="Optional path to write JSON metrics.")

    # Logging cosmetics
    p.add_argument("--no-rich", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _print_human_summary(logger, metrics: dict) -> None:
    mean_r = float(metrics.get("mean_reward", float("nan")))
    std_r = float(metrics.get("std_reward", float("nan")))
    mean_l = float(metrics.get("mean_length", float("nan")))
    win_rate = float(metrics.get("win_rate", 0.0))
    wins = int(metrics.get("wins", 0))
    episodes = int(metrics.get("episodes", 0))
    num_envs = int(metrics.get("num_envs", 1))
    deterministic = bool(metrics.get("deterministic", False))

    logger.info(
        f"[eval] mean_reward={mean_r:.6g} std_reward={std_r:.6g} "
        f"mean_len={mean_l:.3f} win_rate={win_rate:.3f} ({wins}/{episodes}) "
        f"num_envs={num_envs} deterministic={deterministic}"
    )

    tc = metrics.get("termination_counts")
    if isinstance(tc, dict) and tc:
        items = ", ".join([f"{k}={int(v)}" for k, v in sorted(tc.items())])
        logger.info(f"[eval] termination_counts: {items}")

    if "final_score_mean" in metrics:
        logger.info(
            f"[eval] final_score: mean={float(metrics['final_score_mean']):.3f} "
            f"min={float(metrics.get('final_score_min', 0.0)):.3f} "
            f"max={float(metrics.get('final_score_max', 0.0)):.3f}"
        )


def main() -> None:
    args = _parse_args()
    use_rich = not bool(args.no_rich)

    logger = setup_logger(
        name="snake_rl.eval",
        use_rich=use_rich,
        level=str(args.log_level),
    )

    repo = repo_root()
    run_dir = resolve_run_dir(repo, str(args.run))

    cfg, cfg_path = resolve_cli_config(run_dir=run_dir, override=args.config)

    ckpt = pick_checkpoint(run_dir=run_dir, which=str(args.which))
    model: PPO = load_ppo(ckpt, device="auto")

    logger.info(f"[eval] model={ckpt}")
    logger.info(f"[eval] run_dir={run_dir}")
    logger.info(f"[eval] config={cfg_path}")
    logger.info(f"[eval] episodes={int(args.episodes)} deterministic={bool(args.deterministic)}")

    # Default num_envs: from frozen config (run.num_envs), but never exceed episodes
    if args.num_envs is None:
        try:
            cfg_num_envs = int(get_run_num_envs(cfg))
        except Exception:
            cfg_num_envs = 1
        num_envs = max(1, min(int(cfg_num_envs), int(args.episodes)))
    else:
        num_envs = max(1, min(int(args.num_envs), int(args.episodes)))

    logger.info(f"[eval] num_envs={num_envs}")

    paths = _EvalPaths(repo_root=repo)
    log_ppo_params(model=model, cfg=cfg, paths=paths, logger=logger)  # type: ignore[arg-type]

    if args.seed_base is not None:
        seed_base = int(args.seed_base)
    else:
        seed_base = int(get_run_seed(cfg)) + 12_345
    logger.info(f"[eval] seed_base={seed_base}")

    total_eps = int(args.episodes)

    progress = None
    task_id = None

    if use_rich and Progress is not None:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            transient=True,
        )
        progress.start()
        task_id = progress.add_task("eval", total=total_eps)

    def _on_episode(done: int, _total: int, reward: Optional[float]) -> None:
        if progress is None or task_id is None:
            return
        if reward is None:
            return
        progress.update(task_id, completed=int(done), description=f"eval (last_return={float(reward):.3g})")

    try:
        metrics = evaluate_model(
            model=model,
            cfg=cfg,
            episodes=total_eps,
            deterministic=bool(args.deterministic),
            seed_base=int(seed_base),
            num_envs=int(num_envs),
            on_episode=_on_episode if progress is not None else None,
        )
    finally:
        if progress is not None:
            progress.stop()

    metrics["phase"] = "manual"
    metrics["run"] = str(args.run)
    metrics["which"] = str(args.which)
    metrics["run_dir"] = str(run_dir)
    metrics["model_path"] = str(ckpt)
    metrics["config_path"] = str(cfg_path)

    _print_human_summary(logger, metrics)

    if bool(args.json):
        print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        logger.info(f"[eval] wrote: {out_path}")


if __name__ == "__main__":
    main()
