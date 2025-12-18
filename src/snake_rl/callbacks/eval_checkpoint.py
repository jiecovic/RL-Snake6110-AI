# src/snake_rl/callbacks/eval_checkpoint.py
from __future__ import annotations

import json
import time
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback

try:
    # When SB3 progress_bar=True, Rich owns the terminal. Using tqdm.rich prevents
    # raw cursor-control artifacts (e.g. "[A") caused by competing renderers.
    from tqdm.rich import tqdm  # type: ignore
except Exception:  # pragma: no cover
    from tqdm.auto import tqdm

from snake_rl.training.eval_utils import evaluate_model


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_save_zip(
        *,
        model,
        dst: Path,
        retries: int = 25,
        retry_sleep_s: float = 0.05,
) -> None:
    """
    Atomic-ish checkpoint save that behaves well on Windows.

    Strategy:
      1) Save to a temporary file in the same directory.
      2) Atomically replace dst with tmp via os.replace() (same filesystem).
      3) If dst is locked (e.g. snake-watch currently loaded latest.zip), retry.
      4) If still locked, keep training alive by saving a fallback file.

    Important:
      - Do NOT unlink(dst) first. On Windows that often fails if another process holds it open.
      - os.replace() is the right primitive for atomic replacement on both Windows and POSIX.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Keep temp in same directory so replace is atomic (same filesystem)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    # If a stale tmp exists (e.g. after a crash), try to remove it (best effort).
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass

    model.save(str(tmp))

    last_err: Exception | None = None
    for _ in range(max(1, int(retries))):
        try:
            os.replace(str(tmp), str(dst))  # atomic replace (does not require pre-unlink)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(float(retry_sleep_s))
        except OSError as e:
            # Some AV/indexers can cause transient errors too
            last_err = e
            time.sleep(float(retry_sleep_s))

    # If we get here, dst is probably locked. Don't kill training; save a fallback.
    fallback = dst.with_name(dst.stem + f".locked_{int(time.time())}" + dst.suffix)
    try:
        if tmp.exists():
            os.replace(str(tmp), str(fallback))
    except Exception:
        # Best effort cleanup; if even that fails, leave tmp around.
        pass

    raise RuntimeError(
        f"Failed to replace checkpoint {dst} (likely locked by another process). "
        f"Saved fallback checkpoint to {fallback}. Last error: {last_err!r}"
    )


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class EvalCheckpointCallback(BaseCallback):
    """
    Unified callback:
      - saves checkpoints/latest.zip every checkpoint_freq_steps (global env steps)
      - optionally runs intermediate eval (synced with checkpoint cadence)
      - maintains checkpoints/best.zip based on configurable best_metric
      - appends eval history to checkpoints/eval_history.jsonl
      - writes checkpoints/state.json with latest/best metadata

    best_metric:
      - "mean_reward" (default): mean episode return
      - "mean_score": derived from evaluate_model()'s final_score_mean
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
        self.best_path = self.checkpoint_dir / "best.zip"
        self.state_path = self.checkpoint_dir / "state.json"
        self.history_path = self.checkpoint_dir / "eval_history.jsonl"

        self._best_value: Optional[float] = None
        self._best_metric: Optional[str] = None
        self._last_ckpt_at: int = 0

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(Path.cwd()))
        except Exception:
            return str(p)

    def _init_callback(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = _read_json(self.state_path)
        if state and isinstance(state, dict):
            best = state.get("best")
            if isinstance(best, dict):
                m = best.get("metric")
                v = best.get("value")
                if isinstance(m, str):
                    self._best_metric = m
                if v is not None:
                    try:
                        self._best_value = float(v)
                    except Exception:
                        self._best_value = None

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

    def _update_state_best(self, state: dict, *, metric: str, value: float) -> dict:
        state["best"] = {
            "path": self._rel(self.best_path),
            "timesteps": int(self.num_timesteps),
            "metric": str(metric),
            "value": float(value),
            "wall_time": _utc_now_iso(),
        }
        return state

    def _log_eval_to_tb(self, metrics: Dict[str, Any]) -> None:
        self.logger.record("eval/mean_reward", float(metrics["mean_reward"]))
        self.logger.record("eval/std_reward", float(metrics["std_reward"]))
        self.logger.record("eval/mean_length", float(metrics["mean_length"]))
        self.logger.record("eval/std_length", float(metrics["std_length"]))

        if "final_score_mean" in metrics:
            self.logger.record("eval/final_score_mean", float(metrics["final_score_mean"]))
        if "final_score_min" in metrics:
            self.logger.record("eval/final_score_min", float(metrics["final_score_min"]))
        if "final_score_max" in metrics:
            self.logger.record("eval/final_score_max", float(metrics["final_score_max"]))

        term = metrics.get("termination_counts")
        if isinstance(term, dict):
            for k, v in term.items():
                try:
                    self.logger.record(f"eval/termination/{k}", int(v))
                except Exception:
                    pass

    def _pick_best_value(self, metrics: Dict[str, Any], *, best_metric: str) -> float:
        if best_metric == "mean_reward":
            return float(metrics["mean_reward"])
        if best_metric == "mean_score":
            if "final_score_mean" not in metrics:
                raise KeyError(
                    "best_metric='mean_score' requires env to expose info['final_score'] at episode end "
                    "so eval can compute final_score_mean."
                )
            return float(metrics["final_score_mean"])
        raise ValueError("best_metric must be one of: 'mean_reward', 'mean_score'")

    def _on_step(self) -> bool:
        if not self._should_checkpoint_now():
            return True

        self._last_ckpt_at = int(self.num_timesteps)

        _atomic_save_zip(model=self.model, dst=self.latest_path)

        state = _read_json(self.state_path) or {}
        state = self._update_state_latest(state)
        _write_json(self.state_path, state)

        if self.verbose > 0:
            print(f"[ckpt] latest @ {self.num_timesteps}: {self._rel(self.latest_path)}", flush=True)

        phase = getattr(self.cfg, "eval", None)
        intermediate = getattr(phase, "intermediate", None) if phase is not None else None
        if intermediate is None or not bool(getattr(intermediate, "enabled", False)):
            return True

        seed_base = int(self.cfg.run.seed) + int(getattr(intermediate, "seed_offset", 10_000))
        episodes = int(getattr(intermediate, "episodes", 10))
        deterministic = bool(getattr(intermediate, "deterministic", True))
        best_metric = str(getattr(intermediate, "best_metric", "mean_reward"))

        if self.verbose > 0:
            print(
                f"[eval] start intermediate @ {self.num_timesteps}: "
                f"episodes={episodes} deterministic={deterministic} best_metric={best_metric}",
                flush=True,
            )

        pbar: Optional[tqdm] = None
        if self.verbose > 0:
            pbar = tqdm(
                total=episodes,
                desc=f"eval@{self.num_timesteps}",
                leave=False,
                dynamic_ncols=True,
                position=1,
            )

        def _on_episode(i: int, n: int, reward: Optional[float]) -> None:
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

        metrics["phase"] = "intermediate"
        metrics["timesteps"] = int(self.num_timesteps)
        metrics["wall_time"] = _utc_now_iso()
        metrics["best_metric"] = best_metric

        _append_jsonl(self.history_path, metrics)
        self._log_eval_to_tb(metrics)

        chosen_value = self._pick_best_value(metrics, best_metric=best_metric)

        if self.verbose > 0:
            extra = ""
            if "final_score_mean" in metrics:
                extra = f" mean_score={metrics['final_score_mean']:.6g}"
            print(
                f"[eval] done  intermediate @ {self.num_timesteps}: "
                f"mean_reward={metrics['mean_reward']:.6g} std_reward={metrics['std_reward']:.6g} "
                f"mean_len={metrics['mean_length']:.3f}{extra}",
                flush=True,
            )

        is_best = (
                self._best_value is None
                or self._best_metric != best_metric
                or chosen_value > float(self._best_value)
        )

        if is_best:
            self._best_value = float(chosen_value)
            self._best_metric = best_metric
            _atomic_save_zip(model=self.model, dst=self.best_path)

            state = _read_json(self.state_path) or {}
            state = self._update_state_latest(state)
            state = self._update_state_best(state, metric=best_metric, value=chosen_value)
            _write_json(self.state_path, state)

            if self.verbose > 0:
                print(
                    f"[ckpt] best  @ {self.num_timesteps}: {self._rel(self.best_path)} "
                    f"({best_metric}={chosen_value:.6g})",
                    flush=True,
                )
        else:
            state = _read_json(self.state_path) or {}
            state = self._update_state_latest(state)
            _write_json(self.state_path, state)

        return True
