# src/snake_rl/training/train_loop.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from snake_rl.config.schema import TrainConfig
from snake_rl.training.callbacks_factory import make_callbacks
from snake_rl.training.env_factory import make_vec_env
from snake_rl.training.eval_utils import evaluate_model
from snake_rl.training.model_factory import make_or_load_model
from snake_rl.training.resume import resolve_resume_arg
from snake_rl.training.run_paths import RunPaths, make_run_paths


def save_manifest(*, run_dir: Path, cfg: TrainConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(*, cfg: TrainConfig, resume_override: Optional[str] = None) -> RunPaths:
    paths = make_run_paths(run_name=str(cfg.run.name))
    save_manifest(run_dir=paths.run_dir, cfg=cfg)

    set_random_seed(int(cfg.run.seed))

    vec_env = make_vec_env(cfg=cfg)

    resume_path: Optional[Path] = None
    resume_value = resume_override if resume_override is not None else cfg.run.resume_checkpoint
    if resume_value:
        resume_path = resolve_resume_arg(str(resume_value), paths.experiments_root)

    model: PPO = make_or_load_model(
        cfg=cfg,
        vec_env=vec_env,
        tensorboard_log=paths.tb_dir,
        resume_path=resume_path,
    )

    callbacks = make_callbacks(
        cfg=cfg,
        checkpoint_dir=paths.checkpoint_dir,
    )

    print(f"[train] run_id={paths.run_id}")
    print(f"[train] run_dir={paths.run_dir}")
    print(f"[train] tb_dir={paths.tb_dir}")
    print(f"[train] checkpoints={paths.checkpoint_dir}")
    print(f"[train] obs_space={vec_env.observation_space}")
    print(f"[train] action_space={vec_env.action_space}")
    print(f"[train] torch={torch.__version__} cuda={torch.cuda.is_available()}")

    model.learn(
        total_timesteps=int(cfg.run.total_timesteps),
        progress_bar=True,
        callback=callbacks,
        tb_log_name="ppo",
    )

    # Optional: keep an explicit final artifact, even though latest.zip exists.
    final_path = paths.checkpoint_dir / "final.zip"
    model.save(str(final_path))

    # Final "serious" eval (optional, cfg-driven)
    if bool(cfg.eval.final.enabled):
        seed_base = int(cfg.run.seed) + int(cfg.eval.final.seed_offset)
        metrics = evaluate_model(
            model=model,
            cfg=cfg,
            episodes=int(cfg.eval.final.episodes),
            deterministic=bool(cfg.eval.final.deterministic),
            seed_base=seed_base,
        )
        metrics["phase"] = "final"
        metrics["timesteps"] = int(cfg.run.total_timesteps)

        # Write a stable JSON summary + also append to history JSONL
        (paths.run_dir / "eval_final.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _append_jsonl(paths.checkpoint_dir / "eval_history.jsonl", metrics)

        print(
            f"[eval-final] mean_reward={metrics['mean_reward']:.6g} "
            f"std_reward={metrics['std_reward']:.6g} "
            f"mean_len={metrics['mean_length']:.3f}"
        )

    vec_env.close()
    print(f"[train] Done. Final model saved to: {final_path}")
    return paths
