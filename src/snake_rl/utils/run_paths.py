# src/snake_rl/utils/run_paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from snake_rl.utils.paths import repo_root, runs_root


@dataclass(frozen=True)
class RunPaths:
    repo_root: Path
    runs_root: Path
    run_id: str
    run_dir: Path
    tb_dir: Path
    checkpoint_dir: Path


def unique_run_id(base: str, runs_root_path: Path) -> str:
    runs_root_path.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        run_id = f"{base}_{i:03d}"
        if not (runs_root_path / run_id).exists():
            return run_id
        i += 1


def make_run_paths(*, run_name: str) -> RunPaths:
    root = repo_root()

    runs_root_path = runs_root()
    run_id = unique_run_id(run_name, runs_root_path)

    run_dir = runs_root_path / run_id
    tb_dir = run_dir / "tb"
    checkpoint_dir = run_dir / "checkpoints"

    tb_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        repo_root=root,
        runs_root=runs_root_path,
        run_id=run_id,
        run_dir=run_dir,
        tb_dir=tb_dir,
        checkpoint_dir=checkpoint_dir,
    )
