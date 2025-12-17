# src/snake_rl/callbacks/checkpoint.py
from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


class SingleFileCheckpointCallback(BaseCallback):
    """
    Save the model to a single fixed .zip file every N steps (overwrites each time).
    """

    def __init__(
            self,
            *,
            save_freq: int,
            save_path: str | Path,
            filename: str = "latest",
            verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = Path(save_path)
        self.filename = filename

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _model_path(self) -> Path:
        return self.save_path / f"{self.filename}.zip"

    def _on_step(self) -> bool:
        if self.save_freq > 0 and (self.num_timesteps % self.save_freq == 0):
            path = self._model_path()
            self.model.save(str(path))
            if self.verbose >= 1:
                print(f"[checkpoint] saved {path}")
        return True
