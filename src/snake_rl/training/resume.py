# src/snake_rl/training/resume.py
from __future__ import annotations

from pathlib import Path


def _resolve_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """
    Resolve the latest checkpoint in a checkpoint directory.
    Prefers 'latest.zip' if present, otherwise falls back to newest .zip by mtime.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    latest = checkpoint_dir / "latest.zip"
    if latest.is_file():
        return latest.resolve()

    zips = sorted(checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    if not zips:
        raise FileNotFoundError(f"No checkpoint .zip files found in: {checkpoint_dir}")

    return zips[-1].resolve()


def resolve_resume_arg(resume: str, experiments_root: Path) -> Path:
    """
    Accept:
      - path/to/model.zip
      - <run_id>                  -> experiments/<run_id>/checkpoints/latest.zip
      - latest:<run_name_prefix>  -> newest experiments/<run_name_prefix>_*/checkpoints/latest.zip
    """
    p = Path(resume).expanduser()

    # Explicit checkpoint path
    if resume.endswith(".zip") and p.is_file():
        return p.resolve()

    # latest:<run_name_prefix>
    if resume.startswith("latest:"):
        prefix = resume.split("latest:", 1)[1].strip()
        if not prefix:
            raise ValueError("Invalid --resume value. Expected latest:<run_name_prefix>.")

        candidates = sorted(
            [d for d in experiments_root.glob(f"{prefix}_*") if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No experiment runs found in {experiments_root} matching prefix: {prefix!r}"
            )

        newest_run = candidates[-1]
        return _resolve_latest_checkpoint(newest_run / "checkpoints")

    # <run_id>
    run_dir = experiments_root / resume
    if run_dir.is_dir():
        return _resolve_latest_checkpoint(run_dir / "checkpoints")

    raise FileNotFoundError(f"Could not interpret --resume argument: {resume!r}")
