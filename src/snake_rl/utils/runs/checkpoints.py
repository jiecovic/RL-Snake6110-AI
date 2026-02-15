# src/snake_rl/utils/runs/checkpoints.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, cast


def read_json(path: Path) -> dict[str, Any] | None:
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
      - latest/final -> <run_dir>/checkpoints/{name}.zip
      - best/best_reward/best_score/best_win -> named best checkpoint
      - auto -> prefer latest, then best_reward, best_score, best_win
    """
    ckpt_dir = run_dir / "checkpoints"

    if which in ("latest", "final"):
        p = ckpt_dir / f"{which}.zip"
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p
    if which in ("best", "best_reward"):
        p = ckpt_dir / "best_reward.zip"
        if p.is_file():
            return p
        which = "best"
    if which in ("best_score", "best_win"):
        p = ckpt_dir / f"{which}.zip"
        if p.is_file():
            return p

    if which == "auto":
        state = read_json(ckpt_dir / "state.json") or {}
        latest = state.get("latest", {})
        if isinstance(latest, dict):
            path = latest.get("path")
            if isinstance(path, str):
                p = run_dir / path
                if p.is_file():
                    return p

        p = ckpt_dir / "latest.zip"
        if p.is_file():
            return p

        best = state.get("best")
        if isinstance(best, dict):
            for key in ("reward", "score", "win"):
                entry = best.get(key)
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                if isinstance(path, str):
                    p = run_dir / path
                    if p.is_file():
                        return p
            legacy = best.get("path")
            if isinstance(legacy, str):
                p = run_dir / legacy
                if p.is_file():
                    return p

        for name in ("best_reward.zip", "best_score.zip", "best_win.zip", "best.zip"):
            p = ckpt_dir / name
            if p.is_file():
                return p

        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    if which == "best":
        legacy = ckpt_dir / "best.zip"
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir / 'best_reward.zip'}")

    raise ValueError(
        "which must be one of: auto, latest, best, best_reward, best_score, best_win, final"
    )


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
