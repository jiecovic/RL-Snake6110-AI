# src/snake_rl/config/snapshot.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

import yaml

SnapshotConfig = dict[str, Any]


def load_snapshot_config(*, run_dir: Path) -> SnapshotConfig:
    """
    Load the snapshot config from a run directory.

    Single source of truth:
      <run_dir>/config_snapshot.yaml
    """
    cfg_path = Path(run_dir) / "config_snapshot.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Could not find config_snapshot.yaml in run dir: {run_dir}")

    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"config_snapshot.yaml must parse to a dict, got {type(data).__name__}")
    return cast(SnapshotConfig, data)


def load_snapshot_config_path(*, path: Path) -> SnapshotConfig:
    """
    Load a snapshot config from an explicit YAML path.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path.name} must parse to a dict, got {type(data).__name__}")
    return cast(SnapshotConfig, data)


def _get(cfg: Any, path: str, default: Any = None) -> Any:
    """
    Nested access for dict configs only. Path like "run.seed" or "env.id".
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict):
            return default
        if part not in cur:
            return default
        cur = cur[part]
    return cur


def require_int(cfg: SnapshotConfig, path: str) -> int:
    v = _get(cfg, path, None)
    if v is None:
        raise KeyError(f"Config missing {path}")
    return int(v)


def optional_int(cfg: SnapshotConfig, path: str, default: int) -> int:
    v = _get(cfg, path, None)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def get_run_seed(cfg: SnapshotConfig) -> int:
    return require_int(cfg, "run.seed")


def get_run_num_envs(cfg: SnapshotConfig) -> int:
    return require_int(cfg, "run.num_envs")


def get_env_id(cfg: SnapshotConfig) -> str:
    v = _get(cfg, "env.id", None)
    if v is None:
        raise KeyError("Config missing env.id")
    return str(v)


def get_env_params(cfg: SnapshotConfig) -> dict[str, Any]:
    v = _get(cfg, "env.params", None)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise TypeError(f"env.params must be a dict, got {type(v).__name__}")
    return dict(v)


def get_level_params(cfg: SnapshotConfig) -> dict[str, int]:
    h = _get(cfg, "level.height", None)
    w = _get(cfg, "level.width", None)
    f = _get(cfg, "level.food_count", None)
    if h is None or w is None or f is None:
        raise KeyError("Config missing one of: level.height, level.width, level.food_count")
    return {"height": int(h), "width": int(w), "food_count": int(f)}


def get_frame_stack_n(cfg: SnapshotConfig) -> int:
    return max(1, optional_int(cfg, "observation.frame_stack.n_frames", 1))


def resolve_cli_config(*, run_dir: Path, override: Optional[str]) -> tuple[SnapshotConfig, Path]:
    """
    CLI helper:
      - if override provided: load that YAML path
      - else: load <run_dir>/config_snapshot.yaml
    Returns (cfg, cfg_path_used).
    """
    if override:
        p = Path(override)
        return load_snapshot_config_path(path=p), p

    p = Path(run_dir) / "config_snapshot.yaml"
    return load_snapshot_config(run_dir=run_dir), p
