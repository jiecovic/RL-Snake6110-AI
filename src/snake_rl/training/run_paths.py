# src/snake_rl/training/run_paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return p
        if (p / ".git").exists():
            return p
    return None


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    runs_root: Path
    models_root: Path
    run_id: str
    run_dir: Path
    model_dir: Path
    checkpoint_dir: Path


def unique_run_id(base: str, runs_root: Path) -> str:
    runs_root.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        run_id = f"{base}_{i:03d}"
        if not (runs_root / run_id).exists():
            return run_id
        i += 1


def make_run_paths(*, run_name: str) -> RunPaths:
    root = _find_repo_root(Path.cwd()) or Path.cwd().resolve()

    runs_root = root / "runs"
    models_root = root / "models"

    run_id = unique_run_id(run_name, runs_root)

    run_dir = runs_root / run_id
    model_dir = models_root / run_id
    checkpoint_dir = model_dir / "checkpoints"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        repo_root=root,
        runs_root=runs_root,
        models_root=models_root,
        run_id=run_id,
        run_dir=run_dir,
        model_dir=model_dir,
        checkpoint_dir=checkpoint_dir,
    )
