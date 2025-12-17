# src/snake_rl/callbacks/termination_logger.py
from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, List

from stable_baselines3.common.callbacks import BaseCallback


class TerminationCauseLogger(BaseCallback):
    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.cause_counts: DefaultDict[str, int] = defaultdict(int)
        self.episode_count = 0
        self.final_scores: List[float] = []

    def _on_step(self) -> bool:
        infos: List[dict] = self.locals.get("infos", [])

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

        for cause, count in self.cause_counts.items():
            self.logger.record(f"custom/termination_count/{cause}", count)
            self.logger.record(f"custom/termination_freq/{cause}", count / self.episode_count)

        if self.final_scores:
            self.logger.record("custom/final_score/mean", sum(self.final_scores) / len(self.final_scores))
            self.logger.record("custom/final_score/max", max(self.final_scores))
            self.logger.record("custom/final_score/min", min(self.final_scores))

        self.cause_counts.clear()
        self.episode_count = 0
        self.final_scores.clear()
