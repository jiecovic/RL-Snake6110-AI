# src/snake_rl/training/resume.py
from __future__ import annotations

from pathlib import Path


def _resolve_latest_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    zips = sorted(checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise FileNotFoundError(f"No checkpoint .zip files found in: {checkpoint_dir}")
    return zips[-1]


def resolve_resume_arg(resume: str, models_root: Path) -> Path:
    """
    Accept:
      - path/to/model.zip
      - <run_id> (interpreted as models/<run_id>/checkpoints/latest.zip)
      - latest:<run_name> (interpreted as newest checkpoint in models/<run_name>*/checkpoints)
    """
    p = Path(resume).expanduser()

    if resume.endswith(".zip") and p.is_file():
        return p.resolve()

    if resume.startswith("latest:"):
        base = resume.split("latest:", 1)[1].strip()
        if not base:
            raise ValueError("Invalid --resume value. Expected latest:<run_name> with non-empty run name.")

        candidates = sorted([d for d in models_root.glob(f"{base}_*") if d.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"No model runs found in {models_root} matching prefix: {base!r}")

        newest_run = max(candidates, key=lambda d: d.stat().st_mtime)
        return _resolve_latest_checkpoint(newest_run / "checkpoints")

    run_dir = models_root / resume
    if run_dir.is_dir():
        return _resolve_latest_checkpoint(run_dir / "checkpoints")

    raise FileNotFoundError(f"Could not interpret --resume: {resume!r}")
