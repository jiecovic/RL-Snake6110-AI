# src/snake_rl/config/snapshot.py
from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from snake_rl.config.access import (
    cfg_get as _get,
)
from snake_rl.config.access import (
    get_board_params,
    get_env_action,
    get_env_obs,
    get_frame_stack_n,
    get_run_num_envs,
    get_run_seed,
    get_run_vec,
    optional_int,
    require_int,
)

SnapshotConfig = dict[str, Any]


def load_snapshot_config(*, run_dir: Path) -> SnapshotConfig:
    """
    Load the snapshot config from a run directory.

    Single source of truth:
      <run_dir>/config_snapshot.yaml (full validated config)
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


__all__ = [
    "SnapshotConfig",
    "load_snapshot_config",
    "load_snapshot_config_path",
    "_get",
    "require_int",
    "optional_int",
    "get_run_seed",
    "get_run_num_envs",
    "get_env_action",
    "get_run_vec",
    "get_env_obs",
    "get_board_params",
    "get_frame_stack_n",
]
