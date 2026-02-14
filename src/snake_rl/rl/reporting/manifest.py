# src/snake_rl/rl/reporting/manifest.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from snake_rl.config.schema import TrainConfig


def _render_header(lines: list[str]) -> str:
    return "\n".join(f"# {line}" if line else "#" for line in lines) + "\n"


def _write_yaml_text(
    *,
    path: Path,
    body: str,
    header_lines: list[str] | None = None,
    overwrite: bool = True,
) -> None:
    if not overwrite and path.exists():
        return
    text = body
    if not text.endswith("\n"):
        text += "\n"
    if header_lines:
        text = _render_header(header_lines) + text
    path.write_text(text, encoding="utf-8")


def _to_summary_yaml_dict(cfg: TrainConfig) -> dict[str, Any]:
    """
    Build a curated summary of the effective config fields.

    If you add a new config field that should be visible in the summary,
    add it here.
    """
    fe = cfg.feature_extractor

    env_obs_params = dict(cfg.env.obs.params)
    d: dict[str, Any] = {
        "run": {
            "name": cfg.run.name,
            "seed": int(cfg.run.seed),
            "num_envs": int(cfg.run.num_envs),
            "vec": str(cfg.run.vec),
            "total_timesteps": int(cfg.run.total_timesteps),
            "checkpoint": {
                "freq": int(cfg.run.checkpoint.freq),
                "eval": {
                    "enabled": bool(cfg.run.checkpoint.eval.enabled),
                    "episodes": int(cfg.run.checkpoint.eval.episodes),
                    "deterministic": bool(cfg.run.checkpoint.eval.deterministic),
                    "seed_offset": int(cfg.run.checkpoint.eval.seed_offset),
                },
            },
        },
        "board": {
            "height": int(cfg.board.height),
            "width": int(cfg.board.width),
            "food_count": int(cfg.board.food_count),
            "spawn_random_dir": bool(cfg.board.spawn_random_dir),
        },
        "reward": {
            "max_steps_factor": float(cfg.reward.max_steps_factor),
            "win_reward": float(cfg.reward.win_reward),
            "food_reward": float(cfg.reward.food_reward),
            "food_speed_bonus": float(cfg.reward.food_speed_bonus),
            "fatal_penalty": float(cfg.reward.fatal_penalty),
            "step_penalty_scale": float(cfg.reward.step_penalty_scale),
            "timeout_penalty": float(cfg.reward.timeout_penalty),
            "step_progress_pivot": float(cfg.reward.step_progress_pivot),
            "step_progress_weight": float(cfg.reward.step_progress_weight),
        },
        "env": {
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

    return d


def _snapshot_payload(cfg: TrainConfig, validated_cfg: dict[str, Any] | None) -> dict[str, Any]:
    if validated_cfg is not None:
        return dict(validated_cfg)
    return _to_summary_yaml_dict(cfg)


def save_manifest(
    *,
    run_dir: Path,
    cfg: TrainConfig,
    hydra_yaml: str | None = None,
    validated_cfg: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> None:
    """
    Persist the training configuration plus resolved config artifacts.

    Files:
      - config_snapshot.yaml: full validated config (single source of truth).
      - config_summary.yaml: curated subset for quick inspection.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    if hydra_yaml:
        _write_yaml_text(
            path=run_dir / "config_hydra.yaml",
            body=hydra_yaml,
            header_lines=[
                "Hydra-resolved training config (resolved defaults + overrides).",
                "Source of truth for what was composed.",
            ],
            overwrite=overwrite,
        )

    if validated_cfg is not None:
        _write_yaml_text(
            path=run_dir / "config_validated.yaml",
            body=yaml.safe_dump(
                validated_cfg,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            ),
            header_lines=[
                "Validated training config (pydantic model).",
                "Equivalent to what training uses at runtime.",
            ],
            overwrite=overwrite,
        )

    snapshot_payload = _snapshot_payload(cfg, validated_cfg)
    summary_payload = _to_summary_yaml_dict(cfg)

    _write_yaml_text(
        path=run_dir / "config_summary.yaml",
        body=yaml.safe_dump(
            summary_payload,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        header_lines=[
            "Curated summary config for quick inspection.",
            "Not guaranteed to include all fields.",
        ],
        overwrite=overwrite,
    )

    _write_yaml_text(
        path=run_dir / "config_snapshot.yaml",
        body=yaml.safe_dump(
            snapshot_payload,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        ),
        header_lines=[
            "Full validated training config.",
            "Single source of truth for reproducing a run.",
        ],
        overwrite=overwrite,
    )
