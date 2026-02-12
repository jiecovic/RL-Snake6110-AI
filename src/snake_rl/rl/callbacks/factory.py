# src/snake_rl/rl/callbacks/factory.py
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList

from snake_rl.rl.callbacks.eval_checkpoint import EvalCheckpointCallback
from snake_rl.rl.callbacks.termination_logger import TerminationCauseLogger


def make_callbacks(*, cfg, checkpoint_dir: str | Path):
    return CallbackList(
        [
            EvalCheckpointCallback(
                cfg=cfg,
                checkpoint_dir=Path(checkpoint_dir),
                checkpoint_freq_steps=int(cfg.run.checkpoint_freq),
                verbose=1,
            ),
            TerminationCauseLogger(cfg=cfg),
        ]
    )
