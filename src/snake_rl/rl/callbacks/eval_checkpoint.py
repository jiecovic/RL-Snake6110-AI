# src/snake_rl/rl/callbacks/eval_checkpoint.py
from __future__ import annotations

import logging
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from snake_rl.rl.eval.eval_utils import evaluate_model
from snake_rl.rl.eval.table import EvalTablePrinter
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
        self._next_ckpt_at: int = (
            int(self.checkpoint_freq_steps) if self.checkpoint_freq_steps > 0 else 0
        )
        self._table = EvalTablePrinter()
        self._log = logging.getLogger("snake_rl.train")

    def _table_print(self, line: str) -> None:
        if self._log.handlers:
            self._log.info(line)
        else:
            print(line, flush=True)

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
        if self.checkpoint_freq_steps > 0:
            current = int(self.num_timesteps)
            if current <= 0:
                self._next_ckpt_at = int(self.checkpoint_freq_steps)
            else:
                freq = int(self.checkpoint_freq_steps)
                self._next_ckpt_at = ((current // freq) + 1) * freq

    def _should_checkpoint_now(self) -> bool:
        if self.checkpoint_freq_steps <= 0:
            return False
        if self.num_timesteps == self._last_ckpt_at:
            return False
        return self.num_timesteps >= self._next_ckpt_at

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

        return True

    def _on_step(self) -> bool:
        if not self._should_checkpoint_now():
            return True

        self._last_ckpt_at = int(self.num_timesteps)
        while self._next_ckpt_at <= self.num_timesteps:
            self._next_ckpt_at += int(self.checkpoint_freq_steps)

        atomic_save_zip(model=self.model, dst=self.latest_path)

        state = read_json(self.state_path) or {}
        state = self._update_state_latest(state)
        write_json(self.state_path, state)

        run_cfg = getattr(self.cfg, "run", None)
        checkpoint_cfg = getattr(run_cfg, "checkpoint", None) if run_cfg is not None else None
        eval_cfg = getattr(checkpoint_cfg, "eval", None) if checkpoint_cfg is not None else None
        if eval_cfg is None or not bool(getattr(eval_cfg, "enabled", False)):
            return True

        seed_base = int(self.cfg.run.seed) + int(getattr(eval_cfg, "seed_offset", 10_000))
        episodes = int(getattr(eval_cfg, "episodes", 10))
        deterministic = bool(getattr(eval_cfg, "deterministic", True))

        # Table output will show eval rows; avoid separate start/done lines.

        metrics = evaluate_model(
            model=self.model,
            cfg=self.cfg,
            episodes=episodes,
            deterministic=deterministic,
            seed_base=seed_base,
        )

        metrics["phase"] = "periodic"
        metrics["timesteps"] = int(self.num_timesteps)
        metrics["wall_time"] = _utc_now_iso()

        append_jsonl(self.history_path, metrics)
        self._log_eval_to_tb(metrics)

        updated = False
        flags = {"reward": False, "score": False, "win": False}
        for key in ("reward", "score", "win"):
            if self._maybe_update_best(metrics, key=key):
                updated = True
                flags[key] = True

        if not updated:
            state = read_json(self.state_path) or {}
            state = self._update_state_latest(state)
            write_json(self.state_path, state)

        if self.verbose > 0:
            self._table.emit(
                metrics=metrics,
                timesteps=int(self.num_timesteps),
                episodes=int(episodes),
                deterministic=bool(deterministic),
                best_flags=flags,
                printer=self._table_print,
            )

        return True
