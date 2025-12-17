# src/snake_rl/utils/checkpoints.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, cast


def read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def pick_checkpoint(*, run_dir: Path, which: str) -> Path:
    """
    Resolve a checkpoint path within a run directory.

    Supported:
      - latest/best/final -> <run_dir>/checkpoints/{name}.zip
      - auto -> prefer state.json["best"], then ["latest"], else latest.zip
    """
    ckpt_dir = run_dir / "checkpoints"

    if which in ("latest", "best", "final"):
        p = ckpt_dir / f"{which}.zip"
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    if which == "auto":
        state = read_json(ckpt_dir / "state.json") or {}
        for key in ("best", "latest"):
            entry = state.get(key, {})
            path = entry.get("path")
            if isinstance(path, str):
                p = run_dir / path
                if p.is_file():
                    return p

        p = ckpt_dir / "latest.zip"
        if p.is_file():
            return p

        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    raise ValueError("which must be one of: auto, latest, best, final")
