# src/snake_rl/training/train_loop.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from snake_rl.training.callbacks_factory import make_callbacks
from snake_rl.training.env_factory import make_vec_env
from snake_rl.training.model_factory import make_or_load_model
from snake_rl.training.resume import resolve_resume_arg
from snake_rl.training.run_paths import RunPaths, make_run_paths


def save_manifest(*, run_dir: Path, cfg: Any) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")


def train(*, cfg: Any, resume_override: Optional[str] = None) -> RunPaths:
    paths = make_run_paths(run_name=str(cfg.run.name))
    save_manifest(run_dir=paths.run_dir, cfg=cfg)

    set_random_seed(int(cfg.run.seed))

    vec_env = make_vec_env(cfg=cfg)

    resume_path: Optional[Path] = None
    resume_value = resume_override if resume_override is not None else cfg.run.resume_checkpoint
    if resume_value:
        resume_path = resolve_resume_arg(str(resume_value), paths.models_root)

    model: PPO = make_or_load_model(
        cfg=cfg,
        vec_env=vec_env,
        tensorboard_log=paths.run_dir,
        resume_path=resume_path,
    )

    callbacks = make_callbacks(
        checkpoint_dir=str(paths.checkpoint_dir),
        checkpoint_freq_steps=int(cfg.run.checkpoint_freq),
        num_envs=int(cfg.run.num_envs),
    )

    print(f"[train] run_id={paths.run_id}")
    print(f"[train] logs={paths.run_dir}")
    print(f"[train] models={paths.model_dir}")
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

    final_path = paths.model_dir / "ppo_final"
    model.save(str(final_path))
    vec_env.close()
    print(f"[train] Done. Final model saved to: {final_path}")
    return paths
