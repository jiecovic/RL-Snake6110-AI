# src/snake_rl/utils/run_config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def load_config_resolved(*, run_dir: Path) -> dict[str, Any]:
    """
    Load <run_dir>/config_resolved.json and perform minimal sanity checks needed by CLIs.
    """
    cfg_path = run_dir / "config_resolved.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Could not find config_resolved.json in run dir: {run_dir}")

    d = cast(dict[str, Any], json.loads(cfg_path.read_text(encoding="utf-8")))

    # minimal sanity (kept intentionally small; schema validation happens earlier)
    _ = d["env"]["id"]
    _ = d["env"].get("params", {})
    _ = d["level"]["height"]
    _ = d["level"]["width"]
    _ = d["level"]["food_count"]

    return d


def get_env_id(cfg_d: dict[str, Any]) -> str:
    return str(cast(dict[str, Any], cfg_d["env"])["id"])


def get_env_params(cfg_d: dict[str, Any]) -> dict[str, Any]:
    return dict(cast(dict[str, Any], cfg_d["env"]).get("params", {}))


def get_level_params(cfg_d: dict[str, Any]) -> dict[str, int]:
    level_d = cast(dict[str, Any], cfg_d["level"])
    return {
        "height": int(level_d["height"]),
        "width": int(level_d["width"]),
        "food_count": int(level_d["food_count"]),
    }


def get_frame_stack_n(cfg_d: dict[str, Any]) -> int:
    fs = cast(dict[str, Any], cfg_d.get("observation", {})).get("frame_stack", {})
    try:
        n = int(cast(dict[str, Any], fs).get("n_frames", 1))
    except (TypeError, ValueError):
        n = 1
    return max(1, n)
