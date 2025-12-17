# src/snake_rl/utils/paths.py
from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    """
    Walk upwards from `start` and return the first directory that looks like the repo root.
    Heuristics:
      - contains 'pyproject.toml' OR '.git' OR 'assets/'.
    """
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return p
        if (p / ".git").exists():
            return p
        if (p / "assets").is_dir():
            return p
    return None


def repo_root() -> Path:
    """
    Return the repository root. In dev mode this is the project directory.
    If not found, raises with a helpful error.
    """
    env = os.environ.get("SNAKE_RL_REPO_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"SNAKE_RL_REPO_ROOT points to a non-directory: {p}")
        return p

    here = Path(__file__).resolve()
    root = _find_repo_root(here.parent)
    if root is None:
        raise FileNotFoundError(
            "Could not locate repo root. Set SNAKE_RL_REPO_ROOT to your project directory."
        )
    return root


def assets_dir() -> Path:
    """
    Return the assets directory (repo_root/assets), with optional override.
    """
    env = os.environ.get("SNAKE_RL_ASSETS_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"SNAKE_RL_ASSETS_DIR points to a non-directory: {p}")
        return p

    p = repo_root() / "assets"
    if not p.is_dir():
        raise FileNotFoundError(f"Assets directory not found: {p}")
    return p


def asset_path(rel: str) -> Path:
    """
    Resolve an asset path relative to assets_dir().
    Example: asset_path('levels/test_level.yaml')
    """
    p = assets_dir() / rel
    return p
