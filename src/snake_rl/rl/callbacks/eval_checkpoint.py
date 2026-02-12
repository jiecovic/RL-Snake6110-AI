# src/snake_rl/rl/callbacks/eval_checkpoint.py
from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

try:
    # When SB3 progress_bar=True, Rich owns the terminal. Using tqdm.rich prevents
    # raw cursor-control artifacts (e.g. "[A") caused by competing renderers.
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm

from snake_rl.rl.eval.eval_utils import evaluate_model
from snake_rl.rl.metrics import (
    Metrics,
    eval_metric_keys,
    is_termination_metric,
    resolve_log_keys,
    resolve_log_termination,
)
from snake_rl.utils.runs.checkpoints import append_jsonl, atomic_save_zip, read_json, write_json


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EvalCheckpointCallback(BaseCallback):
    """
    Unified callback:
      - saves checkpoints/latest.zip every checkpoint_freq_steps (global env steps)
      - optionally runs periodic eval (synced with checkpoint cadence)
      - maintains best checkpoints per metric:
          * best_reward.zip: episode/return_mean
          * best_score.zip: episode/score_mean (if available)
          * best_win.zip: episode/win_rate (only when wins > 0)
      - appends eval history to checkpoints/eval_history.jsonl
      - writes checkpoints/state.json with latest/best metadata
    """

    def __init__(
        self,
        *,
        cfg: Any,
        checkpoint_dir: Path,
        checkpoint_freq_steps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq_steps = int(checkpoint_freq_steps)

        self.latest_path = self.checkpoint_dir / "latest.zip"
        self.best_paths = {
            "reward": self.checkpoint_dir / "best_reward.zip",
            "score": self.checkpoint_dir / "best_score.zip",
            "win": self.checkpoint_dir / "best_win.zip",
        }
        self.state_path = self.checkpoint_dir / "state.json"
        self.history_path = self.checkpoint_dir / "eval_history.jsonl"

        self._best_values: dict[str, float | None] = {
            "reward": None,
            "score": None,
            "win": None,
        }
        self._last_ckpt_at: int = 0

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(Path.cwd()))
        except Exception:
            return str(p)

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = read_json(self.state_path) or {}
        if isinstance(state, dict):
            best = state.get("best")
            if isinstance(best, dict):
                for key in ("reward", "score", "win"):
                    entry = best.get(key)
                    if not isinstance(entry, dict):
                        continue
                    v = entry.get("value")
                    if v is None:
                        continue
                    try:
                        self._best_values[key] = float(v)
                    except Exception:
                        self._best_values[key] = None

    def _should_checkpoint_now(self) -> bool:
        if self.checkpoint_freq_steps <= 0:
            return False
        if self.num_timesteps == self._last_ckpt_at:
            return False
        return (self.num_timesteps % self.checkpoint_freq_steps) == 0

    def _update_state_latest(self, state: dict) -> dict:
        state["latest"] = {
            "path": self._rel(self.latest_path),
            "timesteps": int(self.num_timesteps),
            "wall_time": _utc_now_iso(),
        }
        return state

    def _update_state_best(self, state: dict, *, metric: str, value: float, path: Path) -> dict:
        best = state.get("best")
        if not isinstance(best, dict):
            best = {}
            state["best"] = best
        best[str(metric)] = {
            "path": self._rel(path),
            "timesteps": int(self.num_timesteps),
            "value": float(value),
            "wall_time": _utc_now_iso(),
        }
        return state

    def _log_eval_to_tb(self, metrics: dict[str, Any]) -> None:
        log_keys = resolve_log_keys(
            self.cfg,
            group="eval",
            default_keys=eval_metric_keys(),
        )
        ordered = [k for k in eval_metric_keys() if k in log_keys]
        extras = sorted(k for k in log_keys if k not in eval_metric_keys())
        for key in (*ordered, *extras):
            if key not in log_keys or key not in metrics:
                continue
            with suppress(Exception):
                if key == Metrics.EP_WINS:
                    self.logger.record(f"eval/{key}", int(metrics[key]))
                else:
                    self.logger.record(f"eval/{key}", float(metrics[key]))

        if resolve_log_termination(self.cfg, group="eval", default=True):
            for key, value in metrics.items():
                if not is_termination_metric(key):
                    continue
                with suppress(Exception):
                    self.logger.record(f"eval/{key}", float(value))

    def _metric_value(self, metrics: dict[str, Any], *, key: str) -> float | None:
        if key == "reward":
            return float(metrics[Metrics.EP_RETURN_MEAN])
        if key == "score":
            if Metrics.EP_SCORE_MEAN not in metrics:
                return None
            return float(metrics[Metrics.EP_SCORE_MEAN])
        if key == "win":
            wins = int(metrics.get(Metrics.EP_WINS, 0))
            if wins <= 0:
                return None
            if Metrics.EP_WIN_RATE not in metrics:
                return None
            return float(metrics[Metrics.EP_WIN_RATE])
        return None

    def _maybe_update_best(self, metrics: dict[str, Any], *, key: str) -> bool:
        value = self._metric_value(metrics, key=key)
        if value is None:
            return False
        prev = self._best_values.get(key)
        is_best = prev is None or float(value) > float(prev)
        if not is_best:
            return False

        path = self.best_paths[key]
        atomic_save_zip(model=self.model, dst=path)
        self._best_values[key] = float(value)

        state = read_json(self.state_path) or {}
        state = self._update_state_latest(state)
        state = self._update_state_best(state, metric=key, value=float(value), path=path)
        write_json(self.state_path, state)

        if self.verbose > 0:
            label = {
                "reward": "mean_reward",
                "score": "mean_score",
                "win": "win_rate",
            }.get(key, key)
            print(
                f"[ckpt] best_{key} @ {self.num_timesteps}: {self._rel(path)} "
                f"({label}={float(value):.6g})",
                flush=True,
            )
        return True

    def _on_step(self) -> bool:
        if not self._should_checkpoint_now():
            return True

        self._last_ckpt_at = int(self.num_timesteps)

        atomic_save_zip(model=self.model, dst=self.latest_path)

        state = read_json(self.state_path) or {}
        state = self._update_state_latest(state)
        write_json(self.state_path, state)

        if self.verbose > 0:
            print(
                f"[ckpt] latest @ {self.num_timesteps}: {self._rel(self.latest_path)}",
                flush=True,
            )

        train_cfg = getattr(self.cfg, "train", None)
        eval_cfg = getattr(train_cfg, "eval", None) if train_cfg is not None else None
        if eval_cfg is None or not bool(getattr(eval_cfg, "enabled", False)):
            return True

        seed_base = int(self.cfg.run.seed) + int(getattr(eval_cfg, "seed_offset", 10_000))
        episodes = int(getattr(eval_cfg, "episodes", 10))
        deterministic = bool(getattr(eval_cfg, "deterministic", True))

        if self.verbose > 0:
            print(
                f"[eval] start @ {self.num_timesteps}: "
                f"episodes={episodes} deterministic={deterministic}",
                flush=True,
            )

        pbar: Any | None = None
        if self.verbose > 0:
            pbar = tqdm(
                total=episodes,
                desc=f"eval@{self.num_timesteps}",
                leave=False,
                dynamic_ncols=True,
                position=1,
            )

        def _on_episode(_i: int, _n: int, reward: float | None) -> None:
            if reward is None:
                return
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"return={float(reward):.6g}", refresh=True)

        try:
            metrics = evaluate_model(
                model=self.model,
                cfg=self.cfg,
                episodes=episodes,
                deterministic=deterministic,
                seed_base=seed_base,
                on_episode=_on_episode,
            )
        finally:
            if pbar is not None:
                pbar.close()

        metrics["phase"] = "periodic"
        metrics["timesteps"] = int(self.num_timesteps)
        metrics["wall_time"] = _utc_now_iso()

        append_jsonl(self.history_path, metrics)
        self._log_eval_to_tb(metrics)

        if self.verbose > 0:
            extra = ""
            if Metrics.EP_SCORE_MEAN in metrics:
                extra = f" mean_score={metrics[Metrics.EP_SCORE_MEAN]:.6g}"
            if Metrics.EP_WIN_RATE in metrics:
                wins = int(metrics.get(Metrics.EP_WINS, 0))
                extra += f" win_rate={metrics[Metrics.EP_WIN_RATE]:.3f} ({wins}/{episodes})"
            print(
                f"[eval] done  @ {self.num_timesteps}: "
                f"mean_reward={metrics[Metrics.EP_RETURN_MEAN]:.6g} "
                f"std_reward={metrics[Metrics.EP_RETURN_STD]:.6g} "
                f"mean_len={metrics[Metrics.EP_LENGTH_MEAN]:.3f}{extra}",
                flush=True,
            )

        updated = False
        for key in ("reward", "score", "win"):
            if self._maybe_update_best(metrics, key=key):
                updated = True

        if not updated:
            state = read_json(self.state_path) or {}
            state = self._update_state_latest(state)
            write_json(self.state_path, state)

        return True
