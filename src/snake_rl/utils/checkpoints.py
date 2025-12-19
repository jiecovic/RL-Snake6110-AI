# src/snake_rl/utils/checkpoints.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional, cast


def read_json(path: Path) -> Optional[dict[str, Any]]:
    """
    Read small checkpoint metadata JSON files (e.g. checkpoints/state.json).
    Returns None if the file does not exist.
    """
    if not path.is_file():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, data: dict[str, Any]) -> None:
    """
    Write checkpoint metadata JSON files (e.g. checkpoints/state.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False), encoding="utf-8")


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """
    Append a single JSON object as a line (e.g. checkpoints/eval_history.jsonl).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def atomic_save_zip(
    *,
    model,
    dst: Path,
    retries: int = 25,
    retry_sleep_s: float = 0.05,
) -> None:
    """
    Windows-safe atomic-ish checkpoint save.

    - Save to tmp in same directory
    - Replace with os.replace (atomic on same filesystem)
    - Retry if dst is locked (e.g. watcher loaded the zip)
    - If still locked, keep tmp as a fallback copy and raise
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp = dst.with_suffix(dst.suffix + ".tmp")

    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass

    model.save(str(tmp))

    last_err: Exception | None = None
    for _ in range(max(1, int(retries))):
        try:
            os.replace(str(tmp), str(dst))
            return
        except PermissionError as e:
            last_err = e
            time.sleep(float(retry_sleep_s))
        except OSError as e:
            last_err = e
            time.sleep(float(retry_sleep_s))

    fallback = dst.with_name(dst.stem + f".locked_{int(time.time())}" + dst.suffix)
    try:
        if tmp.exists():
            os.replace(str(tmp), str(fallback))
    except Exception:
        pass

    raise RuntimeError(
        f"Failed to replace checkpoint {dst} (likely locked). "
        f"Saved fallback checkpoint to {fallback}. Last error: {last_err!r}"
    )
