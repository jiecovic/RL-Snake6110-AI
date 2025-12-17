# src/snake_rl/cli/eval.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from tqdm.auto import tqdm

from snake_rl.config.io import load_config
from snake_rl.training.eval_utils import evaluate_model
from snake_rl.training.resume import resolve_resume_arg
from snake_rl.utils.paths import repo_root


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Snake PPO checkpoint/model.")
    p.add_argument("--config", type=str, default="configs/example.yaml", help="Path to YAML config.")
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


def main() -> None:
    args = _parse_args()
    cfg = load_config(Path(args.config))

    experiments_root = repo_root() / "experiments"
    model_path = resolve_resume_arg(str(args.model), experiments_root)

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
