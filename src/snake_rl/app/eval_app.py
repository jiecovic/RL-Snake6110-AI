# src/snake_rl/app/eval_app.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from stable_baselines3 import PPO

from snake_rl.config.access import get_run_num_envs, get_run_seed
from snake_rl.config.loader import load_train_config_from_path
from snake_rl.rl.eval.eval_utils import evaluate_model
from snake_rl.rl.metrics import Metrics, format_eval_summary
from snake_rl.rl.reporting import log_ppo_params
from snake_rl.utils.logging import setup_logger
from snake_rl.utils.models import load_ppo
from snake_rl.utils.runs.checkpoints import pick_checkpoint
from snake_rl.utils.runs.paths import repo_root, resolve_run_dir

try:
    from stable_baselines3.common.callbacks import tqdm as sb3_tqdm  # type: ignore
except Exception:  # pragma: no cover
    sb3_tqdm = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _EvalPaths:
    repo_root: Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Snake PPO checkpoint/model.")
    p.add_argument(
        "--run",
        type=str,
        required=True,
        help="Run id / run folder name under runs/ (legacy: experiments/)",
    )
    p.add_argument(
        "--which",
        type=str,
        default="latest",
        choices=["auto", "latest", "best", "best_reward", "best_score", "best_win", "final"],
    )
    p.add_argument("--episodes", type=int, default=20)

    # None => default from config_snapshot.yaml (env.num_envs), clamped to <= episodes
    p.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Parallel eval envs. Default: env.num_envs from config_snapshot.yaml.",
    )

    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Default: cfg.run.seed + 12345 (from config_snapshot.yaml).",
    )

    # Config resolution: snapshot YAML by default, or Hydra config + overrides
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional config path. If omitted, uses runs/<run>/config_snapshot.yaml. "
            "If path is under configs/, Hydra defaults + overrides are applied."
        ),
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable). Only used when --config is under configs/.",
    )

    # Output controls
    p.add_argument("--json", action="store_true", help="Print full JSON metrics to stdout.")
    p.add_argument("--out", type=str, default=None, help="Optional path to write JSON metrics.")

    # Logging cosmetics
    p.add_argument("--no-rich", action="store_true")
    p.add_argument("--log-level", type=str, default="INFO")
    return p.parse_args()


def _print_human_summary(logger, metrics: dict) -> None:
    logger.info(format_eval_summary(metrics))

    tc = metrics.get("termination_counts")
    if isinstance(tc, dict) and tc:
        items = ", ".join([f"{k}={int(v)}" for k, v in sorted(tc.items())])
        logger.info(f"[eval] termination_counts: {items}")

    if Metrics.EP_SCORE_MEAN in metrics:
        logger.info(
            f"[eval] score: mean={float(metrics[Metrics.EP_SCORE_MEAN]):.3f} "
            f"min={float(metrics.get(Metrics.EP_SCORE_MIN, 0.0)):.3f} "
            f"max={float(metrics.get(Metrics.EP_SCORE_MAX, 0.0)):.3f}"
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

    config_path = Path(args.config) if args.config else (run_dir / "config_snapshot.yaml")
    cfg, _, _, _ = load_train_config_from_path(
        config_path=config_path,
        config_root=repo / "configs",
        overrides=list(args.override),
    )
    cfg_path = config_path

    ckpt = pick_checkpoint(run_dir=run_dir, which=str(args.which))
    model: PPO = load_ppo(ckpt, device="auto")

    logger.info(f"[eval] model={ckpt}")
    logger.info(f"[eval] run_dir={run_dir}")
    logger.info(f"[eval] config={cfg_path}")
    logger.info(f"[eval] episodes={int(args.episodes)} deterministic={bool(args.deterministic)}")

    # Default num_envs: from snapshot config (env.num_envs), but never exceed episodes
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

    pbar = None

    if use_rich and sb3_tqdm is not None:
        pbar = sb3_tqdm(total=total_eps, desc="eval", leave=False)

    _sum_return = 0.0
    _count_return = 0

    sum_steps = 0
    start_ts = perf_counter()

    def _on_episode(done: int, _total: int, reward: float | None, length: int | None) -> None:
        if pbar is None:
            return
        if reward is None:
            return
        nonlocal _sum_return, _count_return
        nonlocal sum_steps
        _sum_return += float(reward)
        _count_return += 1
        mean_return = _sum_return / max(_count_return, 1)
        if length is not None:
            sum_steps += int(length)
        elapsed = perf_counter() - start_ts
        steps_per_s = None if elapsed <= 0 else float(sum_steps) / float(elapsed)
        if steps_per_s is None:
            pbar.set_description(f"eval (mean_return={mean_return:.3g})")
        else:
            pbar.set_description(
                f"eval (mean_return={mean_return:.3g}, steps/s={steps_per_s:,.0f})"
            )
        pbar.update(int(done) - pbar.n)

    try:
        metrics = evaluate_model(
            model=model,
            cfg=cfg,
            episodes=total_eps,
            deterministic=bool(args.deterministic),
            seed_base=int(seed_base),
            num_envs=int(num_envs),
            on_episode=_on_episode if pbar is not None else None,
        )
    finally:
        if pbar is not None:
            pbar.refresh()
            pbar.close()

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
