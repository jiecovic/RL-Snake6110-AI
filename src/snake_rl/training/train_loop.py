# src/snake_rl/training/train_loop.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from snake_rl.config.schema import TrainConfig
from snake_rl.training.callbacks_factory import make_callbacks
from snake_rl.training.env_factory import make_vec_env
from snake_rl.training.eval_utils import evaluate_model
from snake_rl.training.model_factory import make_or_load_model
from snake_rl.training.resume import resolve_resume_arg
from snake_rl.training.run_paths import RunPaths, make_run_paths


def _to_effective_yaml_dict(cfg: TrainConfig) -> dict[str, Any]:
    """
    Convert the parsed TrainConfig (dataclass schema) back into the input YAML schema.

    Motivation:
    - config_resolved.json is a normalized dataclass dump (machine-friendly).
    - config_effective.yaml should be rerunnable as an input config (human-friendly).
    """
    d: dict[str, Any] = {
        "run": {
            "name": cfg.run.name,
            "seed": int(cfg.run.seed),
            "num_envs": int(cfg.run.num_envs),
            "total_timesteps": int(cfg.run.total_timesteps),
            "checkpoint_freq": int(cfg.run.checkpoint_freq),
        },
        "level": {
            "height": int(cfg.level.height),
            "width": int(cfg.level.width),
            "food_count": int(cfg.level.food_count),
        },
        "env": {
            "id": str(cfg.env.id),
        },
        "observation": {
            "params": dict(cfg.observation.params),
        },
        "model": {
            "features_extractor": {
                "cnn": {
                    "type": str(cfg.model.cnn.type),
                    "features_dim": int(cfg.model.cnn.features_dim),
                }
            },
            "net_arch": [int(x) for x in cfg.model.net_arch],
        },
        "ppo": {
            "n_steps": int(cfg.ppo.n_steps),
            "batch_size": int(cfg.ppo.batch_size),
            "n_epochs": int(cfg.ppo.n_epochs),
            "gamma": float(cfg.ppo.gamma),
            "ent_coef": float(cfg.ppo.ent_coef),
            "learning_rate": float(cfg.ppo.learning_rate),
            "verbose": int(cfg.ppo.verbose),
        },
    }

    if cfg.run.resume_checkpoint is not None:
        d["run"]["resume_checkpoint"] = str(cfg.run.resume_checkpoint)

    # Always write eval (even if defaults) so the run artifact is explicit/reproducible.
    d["eval"] = {
        "intermediate": {
            "enabled": bool(cfg.eval.intermediate.enabled),
            "episodes": int(cfg.eval.intermediate.episodes),
            "deterministic": bool(cfg.eval.intermediate.deterministic),
            "seed_offset": int(cfg.eval.intermediate.seed_offset),
        },
        "final": {
            "enabled": bool(cfg.eval.final.enabled),
            "episodes": int(cfg.eval.final.episodes),
            "deterministic": bool(cfg.eval.final.deterministic),
            "seed_offset": int(cfg.eval.final.seed_offset),
        },
    }

    return d


def save_manifest(*, run_dir: Path, cfg: TrainConfig) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # Machine-friendly normalized dump (dataclass schema).
    (run_dir / "config_resolved.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    # Human-friendly rerunnable config (input YAML schema).
    (run_dir / "config_effective.yaml").write_text(
        yaml.safe_dump(
            _to_effective_yaml_dict(cfg),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
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
