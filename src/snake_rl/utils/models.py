# src/snake_rl/utils/models.py
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO


def load_ppo(model_path: Path, *, device: str) -> PPO:
    """
    Load a PPO checkpoint from disk.

    Centralized here so CLI scripts (watch/eval/finetune) don't re-implement
    SB3 load conventions.
    """
    return PPO.load(str(model_path), device=str(device))
