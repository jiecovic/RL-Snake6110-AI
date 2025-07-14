import os
from stable_baselines3.common.callbacks import BaseCallback

class SingleFileCheckpointCallback(BaseCallback):
    """
    Callback for saving a model to a single checkpoint file every ``save_freq`` steps.

    This version overwrites the same file (no incremental suffix) each time.

    :param save_freq: Frequency (in steps) to save the model.
    :param save_path: Directory to save the checkpoint.
    :param filename: Filename to use (without extension).
    :param save_replay_buffer: Whether to save the model replay buffer too.
    :param save_vecnormalize: Whether to save VecNormalize stats.
    :param verbose: Verbosity level.
    """
    def __init__(
        self,
        save_freq: int,
        save_path: str,
        filename: str = "model_checkpoint",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.filename = filename
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        """Return the fixed checkpoint path (no step index)."""
        return os.path.join(self.save_path, f"{self.filename}_{checkpoint_type}.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"✅ Saved model to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                rb_path = self._checkpoint_path("replay_buffer", "pkl")
                self.model.save_replay_buffer(rb_path)
                if self.verbose >= 2:
                    print(f"🧠 Saved replay buffer to {rb_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                vec_path = self._checkpoint_path("vecnormalize", "pkl")
                self.model.get_vec_normalize_env().save(vec_path)
                if self.verbose >= 2:
                    print(f"📊 Saved VecNormalize stats to {vec_path}")

        return True
