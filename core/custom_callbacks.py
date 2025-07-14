from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
from typing import List


class TerminationCauseLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.cause_counts = defaultdict(int)
        self.episode_count = 0
        self.total_final_score = 0.0
        self.final_scores = []

    def _on_step(self) -> bool:
        infos: List[dict] = self.locals.get("infos", [])

        for info in infos:
            if "termination_cause" in info:
                cause = info["termination_cause"]
                self.cause_counts[cause] += 1
                self.episode_count += 1

                if "final_score" in info:
                    score = info["final_score"]
                    self.final_scores.append(score)

        return True

    def _on_rollout_end(self) -> None:
        if self.episode_count == 0:
            return  # Avoid division by zero

        # Log termination cause stats
        for cause, count in self.cause_counts.items():
            self.logger.record(f"custom/termination_count/{cause}", count)
            self.logger.record(f"custom/termination_freq/{cause}", count / self.episode_count)

        # Log final score statistics
        if self.final_scores:
            self.logger.record("custom/final_score/mean", sum(self.final_scores) / len(self.final_scores))
            self.logger.record("custom/final_score/max", max(self.final_scores))
            self.logger.record("custom/final_score/min", min(self.final_scores))

        # Reset counters
        self.cause_counts.clear()
        self.episode_count = 0
        self.final_scores.clear()
