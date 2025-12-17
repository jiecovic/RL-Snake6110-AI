# src/snake_rl/cli/eval.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO

try:
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm

from snake_rl.config.io import load_config
from snake_rl.training.eval_utils import evaluate_model
from snake_rl.training.resume import resolve_resume_arg
from snake_rl.utils.paths import repo_root


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Snake PPO checkpoint/model.")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to YAML config. If omitted and --model points to a run under experiments/, "
            "we will prefer experiments/<run_id>/config_effective.yaml."
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model .zip path OR <run_id> OR 'latest:<run_name_prefix>' (resolved under repo/experiments). "
            "For <run_id> and latest:<...>, this selects checkpoints/latest.zip."
        ),
    )
    p.add_argument("--episodes", type=int, default=20, help="Number of eval episodes.")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions.")
    p.add_argument(
        "--best-metric",
        type=str,
        default="mean_reward",
        help="mean_reward or mean_score (requires final_score in env info).",
    )
    p.add_argument("--seed", type=int, default=None, help="Override eval seed_base (else cfg.run.seed + 12345).")
    p.add_argument("--out", type=str, default=None, help="Optional path to write JSON metrics.")
    return p.parse_args()


def _infer_run_dir_from_model_path(model_path: Path) -> Optional[Path]:
    # If model_path looks like: experiments/<run_id>/checkpoints/<name>.zip
    # then run_dir is parent of checkpoints.
    try:
        if model_path.parent.name == "checkpoints":
            return model_path.parent.parent
    except Exception:
        pass
    return None


def _pick_config_path(args_config: Optional[str], *, model_path: Path) -> Path:
    if args_config:
        return Path(args_config)

    run_dir = _infer_run_dir_from_model_path(model_path)
    if run_dir is not None:
        eff = run_dir / "config_effective.yaml"
        if eff.is_file():
            return eff

    # Safe fallback for standalone eval usage
    return Path("configs/example.yaml")


def main() -> None:
    args = _parse_args()

    experiments_root = repo_root() / "experiments"
    model_path = resolve_resume_arg(str(args.model), experiments_root)

    cfg_path = _pick_config_path(args.config, model_path=model_path)
    cfg = load_config(cfg_path)

    model = PPO.load(str(model_path))

    seed_base = int(args.seed) if args.seed is not None else int(cfg.run.seed) + 12_345
    best_metric = str(args.best_metric)

    pbar = tqdm(
        total=int(args.episodes),
        desc="eval",
        leave=False,
        dynamic_ncols=True,
    )

    def _on_episode(i: int, n: int, reward):
        if reward is None:
            return
        pbar.update(1)
        pbar.set_postfix_str(f"return={float(reward):.6g}", refresh=True)

    try:
        metrics = evaluate_model(
            model=model,
            cfg=cfg,
            episodes=int(args.episodes),
            deterministic=bool(args.deterministic),
            seed_base=seed_base,
            on_episode=_on_episode,
        )
    finally:
        pbar.close()

    if best_metric == "mean_score":
        if "final_score_mean" not in metrics:
            raise KeyError(
                "best_metric='mean_score' requires env to expose info['final_score'] at episode end "
                "so eval can compute final_score_mean."
            )
        best_value = float(metrics["final_score_mean"])
    elif best_metric == "mean_reward":
        best_value = float(metrics["mean_reward"])
    else:
        raise ValueError("--best-metric must be mean_reward or mean_score")

    metrics["phase"] = "manual"
    metrics["model_path"] = str(model_path)
    metrics["config_path"] = str(cfg_path)
    metrics["best_metric"] = best_metric
    metrics["best_value"] = best_value

    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
