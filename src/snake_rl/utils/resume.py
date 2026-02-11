# src\snake_rl\utils\resume.py
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


def _iter_candidate_runs(root: Path, prefix: str | None) -> list[Path]:
    if prefix:
        return [d for d in root.glob(f"{prefix}_*") if d.is_dir()]
    if root.is_dir():
        return [d for d in root.iterdir() if d.is_dir()]
    return []


def resolve_resume_arg(
    resume: str,
    *,
    runs_root: Path,
    legacy_root: Path | None = None,
) -> Path:
    """
    Accept:
      - path/to/model.zip
      - <run_id>                  -> runs/<run_id>/checkpoints/latest.zip (fallback legacy_root)
      - latest:<run_name_prefix>  -> newest runs/<run_name_prefix>_*/checkpoints/latest.zip
                                    (fallback legacy_root)
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

        candidates = _iter_candidate_runs(runs_root, prefix)
        if legacy_root is not None:
            candidates.extend(_iter_candidate_runs(legacy_root, prefix))

        if not candidates:
            raise FileNotFoundError(f"No runs found in {runs_root} matching prefix: {prefix!r}")

        newest_run = sorted(candidates, key=lambda d: d.stat().st_mtime)[-1]
        return _resolve_latest_checkpoint(newest_run / "checkpoints")

    # <run_id>
    run_dir = runs_root / resume
    if run_dir.is_dir():
        return _resolve_latest_checkpoint(run_dir / "checkpoints")

    if legacy_root is not None:
        legacy_run = legacy_root / resume
        if legacy_run.is_dir():
            return _resolve_latest_checkpoint(legacy_run / "checkpoints")

    raise FileNotFoundError(f"Could not interpret --resume argument: {resume!r}")
