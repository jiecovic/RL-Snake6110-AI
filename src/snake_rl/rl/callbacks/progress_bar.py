# src/snake_rl/rl/callbacks/progress_bar.py
from __future__ import annotations

from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

try:
    from stable_baselines3.common.callbacks import tqdm as sb3_tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    sb3_tqdm = None


class SnakeProgressBarCallback(BaseCallback):
    """
    Progress bar that shows absolute progress when resuming.

    If the model has prior timesteps, the bar starts at current
    and counts toward current + remaining.
    """

    def __init__(self) -> None:
        super().__init__()
        if sb3_tqdm is None:
            raise ImportError("Progress bar requires tqdm+rich (stable-baselines3[extra]).")
        self._tqdm: Any = sb3_tqdm
        self._pbar = None

    def _on_training_start(self) -> None:
        total = int(self.locals.get("total_timesteps", 0))
        current = int(getattr(self.model, "num_timesteps", 0))
        total_steps = max(total, 0)
        initial = min(current, total_steps)
        self._pbar = self._tqdm(total=total_steps)
        if initial > 0:
            # tqdm.rich does not always honor `initial` in the display, so set explicitly.
            self._pbar.n = initial
            self._pbar.last_print_n = initial
            self._pbar.refresh()

    def _on_step(self) -> bool:
        if self._pbar is not None:
            self._pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self) -> None:
        if self._pbar is not None:
            self._pbar.refresh()
            self._pbar.close()
            self._pbar = None
