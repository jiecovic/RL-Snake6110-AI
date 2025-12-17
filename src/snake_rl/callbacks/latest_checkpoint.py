# src/snake_rl/training/callbacks/latest_checkpoint.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback


class LatestCheckpointCallback(BaseCallback):
    """
    Periodically overwrite a single 'latest.zip' checkpoint to avoid spamming disk.

    save_every_steps: counts in "callback calls", i.e. vectorized env steps.
    So if you want "every X total env steps", pass X // num_envs.
    """

    def __init__(self, *, save_path: Path, save_every_steps: int, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.save_path = Path(save_path)
        self.save_every_steps = int(save_every_steps)
        self._calls = 0

    def _init_callback(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        self._calls += 1
        if self.save_every_steps <= 0:
            return True
        if (self._calls % self.save_every_steps) != 0:
            return True

        tmp = self.save_path.with_suffix(".zip.tmp")
        # Save to tmp then replace for atomic-ish behavior on Windows
        self.model.save(str(tmp))
        if self.save_path.exists():
            self.save_path.unlink()
        tmp.replace(self.save_path)

        if self.verbose > 0:
            print(f"[ckpt] wrote latest: {self.save_path}")
        return True
