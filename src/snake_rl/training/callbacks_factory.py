# src/snake_rl/training/callbacks_factory.py
from __future__ import annotations

from stable_baselines3.common.callbacks import CallbackList

from snake_rl.callbacks.latest_checkpoint import LatestCheckpointCallback
from snake_rl.callbacks.termination_logger import TerminationCauseLogger


def make_callbacks(*, checkpoint_dir: str, checkpoint_freq_steps: int, num_envs: int):
    save_freq = max(1, int(checkpoint_freq_steps) // int(num_envs))

    return CallbackList(
        [
            # Always overwrite <checkpoint_dir>/latest.zip
            LatestCheckpointCallback(
                save_path=f"{checkpoint_dir}/latest.zip",
                save_every_steps=save_freq,
            ),
            TerminationCauseLogger(),
        ]
    )
