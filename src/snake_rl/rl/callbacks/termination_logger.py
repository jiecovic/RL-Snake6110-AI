# src/snake_rl/rl/callbacks/termination_logger.py
from __future__ import annotations

from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback

from snake_rl.rl.metrics import (
    Metrics,
    resolve_log_keys,
    resolve_log_termination,
    train_metric_keys,
)


class TerminationCauseLogger(BaseCallback):
    def __init__(self, *, cfg=None, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.cause_counts: defaultdict[str, int] = defaultdict(int)
        self.episode_count = 0
        self.final_scores: list[float] = []

    def _on_step(self) -> bool:
        infos: list[dict] = self.locals.get("infos", [])

        for info in infos:
            if "termination_cause" not in info:
                continue

            cause = str(info["termination_cause"])
            self.cause_counts[cause] += 1
            self.episode_count += 1

            if "final_score" in info:
                self.final_scores.append(float(info["final_score"]))

        return True

    def _on_rollout_end(self) -> None:
        if self.episode_count == 0:
            return

        log_keys = resolve_log_keys(
            self.cfg,
            group="train",
            default_keys=train_metric_keys(),
        )

        if resolve_log_termination(self.cfg, group="train", default=True):
            for cause, count in self.cause_counts.items():
                self.logger.record(f"{Metrics.TERM_PREFIX}{cause}_count", count)
                self.logger.record(f"{Metrics.TERM_PREFIX}{cause}_rate", count / self.episode_count)

        if self.final_scores:
            mean_score = sum(self.final_scores) / len(self.final_scores)
            if Metrics.EP_SCORE_MEAN in log_keys:
                self.logger.record(f"{Metrics.EP_SCORE_MEAN}", mean_score)
            if Metrics.EP_SCORE_MAX in log_keys:
                self.logger.record(f"{Metrics.EP_SCORE_MAX}", max(self.final_scores))
            if Metrics.EP_SCORE_MIN in log_keys:
                self.logger.record(f"{Metrics.EP_SCORE_MIN}", min(self.final_scores))

        wins = int(self.cause_counts.get("win", 0))
        if Metrics.EP_WINS in log_keys:
            self.logger.record(f"{Metrics.EP_WINS}", wins)
        if Metrics.EP_WIN_RATE in log_keys:
            self.logger.record(f"{Metrics.EP_WIN_RATE}", wins / self.episode_count)

        self.cause_counts.clear()
        self.episode_count = 0
        self.final_scores.clear()
