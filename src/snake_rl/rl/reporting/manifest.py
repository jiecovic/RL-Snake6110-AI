# src/snake_rl/rl/reporting/manifest.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from snake_rl.config.schema import TrainConfig


def _to_snapshot_yaml_dict(cfg: TrainConfig) -> dict[str, Any]:
    """
    Build the persisted run snapshot dict.

    Note: This is a curated snapshot of the effective config fields that matter for
    reproducibility across train/eval/watch. If you add a new config field that should
    be reproducible, add it here.
    """
    fe = cfg.feature_extractor

    env_obs_params = dict(cfg.env.obs.params)
    d: dict[str, Any] = {
        "run": {
            "name": cfg.run.name,
            "seed": int(cfg.run.seed),
            "num_envs": int(cfg.run.num_envs),
            "total_timesteps": int(cfg.run.total_timesteps),
            "checkpoint_freq": int(cfg.run.checkpoint_freq),
        },
        "board": {
            "height": int(cfg.board.height),
            "width": int(cfg.board.width),
            "food_count": int(cfg.board.food_count),
        },
        "reward": {
            "max_steps_factor": float(cfg.reward.max_steps_factor),
            "win_reward": float(cfg.reward.win_reward),
            "food_reward": float(cfg.reward.food_reward),
            "food_speed_bonus": float(cfg.reward.food_speed_bonus),
            "fatal_penalty": float(cfg.reward.fatal_penalty),
            "step_penalty_scale": float(cfg.reward.step_penalty_scale),
            "timeout_penalty": float(cfg.reward.timeout_penalty),
        },
        "env": {
            "engine": str(cfg.env.engine),
            "action": {"type": str(cfg.env.action.type)},
            "obs": {
                "kind": str(cfg.env.obs.kind),
                "view": str(cfg.env.obs.view),
                "params": env_obs_params,
                "features": dict(cfg.env.obs.features),
            },
            "frame_stack": {
                "n_frames": int(cfg.env.frame_stack.n_frames),
            },
        },
        "feature_extractor": {
            "type": str(fe.type),
            "features_dim": int(fe.features_dim),
            "params": dict(fe.params),
        },
        "train": {
            "algo": {
                "type": str(cfg.train.algo.type),
                "params": dict(cfg.train.algo.params),
            },
            "eval": {
                "enabled": bool(cfg.train.eval.enabled),
                "episodes": int(cfg.train.eval.episodes),
                "deterministic": bool(cfg.train.eval.deterministic),
                "seed_offset": int(cfg.train.eval.seed_offset),
            },
        },
        "metrics": {
            "eval": {
                "keys": list(cfg.metrics.eval.keys),
                "termination": bool(cfg.metrics.eval.termination),
            },
            "train": {
                "keys": list(cfg.metrics.train.keys),
                "termination": bool(cfg.metrics.train.termination),
            },
        },
    }

    if cfg.run.resume_checkpoint is not None:
        d["run"]["resume_checkpoint"] = str(cfg.run.resume_checkpoint)

    return d


def save_manifest(
    *,
    run_dir: Path,
    cfg: TrainConfig,
    hydra_yaml: str | None = None,
    validated_cfg: dict[str, Any] | None = None,
) -> None:
    """
    Persist the snapshot training configuration plus resolved config artifacts.

    config_snapshot.yaml is the single source of truth for reproducing a run.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    if hydra_yaml:
        (run_dir / "config_hydra.yaml").write_text(hydra_yaml, encoding="utf-8")

    if validated_cfg is not None:
        (run_dir / "config_validated.yaml").write_text(
            yaml.safe_dump(
                validated_cfg,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(
            _to_snapshot_yaml_dict(cfg),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
