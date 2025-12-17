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
    experiments_root: Path
    run_id: str
    run_dir: Path
    tb_dir: Path
    checkpoint_dir: Path


def unique_run_id(base: str, experiments_root: Path) -> str:
    experiments_root.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        run_id = f"{base}_{i:03d}"
        if not (experiments_root / run_id).exists():
            return run_id
        i += 1


def make_run_paths(*, run_name: str) -> RunPaths:
    root = _find_repo_root(Path.cwd()) or Path.cwd().resolve()

    experiments_root = root / "experiments"
    run_id = unique_run_id(run_name, experiments_root)

    run_dir = experiments_root / run_id
    tb_dir = run_dir / "tb"
    checkpoint_dir = run_dir / "checkpoints"

    tb_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        repo_root=root,
        experiments_root=experiments_root,
        run_id=run_id,
        run_dir=run_dir,
        tb_dir=tb_dir,
        checkpoint_dir=checkpoint_dir,
    )
