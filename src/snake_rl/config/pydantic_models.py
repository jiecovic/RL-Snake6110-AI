# src/snake_rl/config/pydantic_models.py

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from snake_rl.config.schema import (
    ActionConfig,
    AlgoConfig,
    BoardConfig,
    EnvConfig,
    EvalConfig,
    FeaturesExtractorConfig,
    FrameStackConfig,
    ObservationConfig,
    RewardConfig,
    RunConfig,
    TrainConfig,
    TrainLoopConfig,
)


class _BaseConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfigModel(_BaseConfigModel):
    name: str
    seed: int
    num_envs: int
    total_timesteps: int
    checkpoint_freq: int
    resume_checkpoint: str | None = None


class BoardConfigModel(_BaseConfigModel):
    height: int
    width: int
    food_count: int


class RewardConfigModel(_BaseConfigModel):
    max_steps_factor: float = 1.3
    win_reward: float = 5.0
    food_reward: float = 2.5
    food_speed_bonus: float = 1.0
    fatal_penalty: float = 5.0
    step_penalty_scale: float = 1.0
    timeout_penalty: float = 0.0


class ActionConfigModel(_BaseConfigModel):
    type: str = "relative"


class ObservationConfigModel(_BaseConfigModel):
    kind: str
    view: str
    params: dict[str, Any] = Field(default_factory=dict)
    features: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind")
    @classmethod
    def _normalize_kind(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s in {"pixel", "pixels"}:
            return "pixel"
        if s in {"categorical", "cat"}:
            return "categorical"
        return str(v)


class FrameStackConfigModel(_BaseConfigModel):
    n_frames: int = 1

    @field_validator("n_frames")
    @classmethod
    def _n_frames_at_least_one(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError("env.frame_stack.n_frames must be >= 1")
        return int(v)


class EnvConfigModel(_BaseConfigModel):
    engine: str = "python"
    action: ActionConfigModel = Field(default_factory=ActionConfigModel)
    obs: ObservationConfigModel
    frame_stack: FrameStackConfigModel = Field(default_factory=FrameStackConfigModel)


class FeaturesExtractorConfigModel(_BaseConfigModel):
    type: str
    features_dim: int
    params: dict[str, Any] = Field(default_factory=dict)


class AlgoConfigModel(_BaseConfigModel):
    type: str = "ppo"
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data, "params": {}}
        if not isinstance(data, dict):
            return data

        if "type" in data and "params" not in data:
            extra = {k: v for k, v in data.items() if k != "type"}
            if extra:
                return {"type": data.get("type"), "params": extra}
        if "params" in data:
            params = data.get("params")
            if params is None:
                params = {}
            if isinstance(params, dict):
                return {"type": data.get("type", "ppo"), "params": dict(params)}
        return data


class EvalConfigModel(_BaseConfigModel):
    enabled: bool = False
    episodes: int = 10
    best_metric: str = "mean_reward"
    deterministic: bool = True
    seed_offset: int = 10_000

    @field_validator("best_metric")
    @classmethod
    def _best_metric_allowed(cls, v: str) -> str:
        s = str(v)
        if s not in {"mean_reward", "mean_score"}:
            raise ValueError("eval.best_metric must be one of: mean_reward, mean_score")
        return s


class TrainLoopConfigModel(_BaseConfigModel):
    algo: AlgoConfigModel = Field(default_factory=AlgoConfigModel)
    eval: EvalConfigModel = Field(default_factory=EvalConfigModel)


class TrainConfigModel(_BaseConfigModel):
    run: RunConfigModel
    board: BoardConfigModel
    reward: RewardConfigModel = Field(default_factory=RewardConfigModel)
    env: EnvConfigModel
    feature_extractor: FeaturesExtractorConfigModel
    train: TrainLoopConfigModel = Field(default_factory=TrainLoopConfigModel)

    def to_dataclass(self) -> TrainConfig:
        return TrainConfig(
            run=RunConfig(
                name=self.run.name,
                seed=int(self.run.seed),
                num_envs=int(self.run.num_envs),
                total_timesteps=int(self.run.total_timesteps),
                checkpoint_freq=int(self.run.checkpoint_freq),
                resume_checkpoint=self.run.resume_checkpoint,
            ),
            board=BoardConfig(
                height=int(self.board.height),
                width=int(self.board.width),
                food_count=int(self.board.food_count),
            ),
            reward=RewardConfig(
                max_steps_factor=float(self.reward.max_steps_factor),
                win_reward=float(self.reward.win_reward),
                food_reward=float(self.reward.food_reward),
                food_speed_bonus=float(self.reward.food_speed_bonus),
                fatal_penalty=float(self.reward.fatal_penalty),
                step_penalty_scale=float(self.reward.step_penalty_scale),
                timeout_penalty=float(self.reward.timeout_penalty),
            ),
            env=EnvConfig(
                engine=str(self.env.engine),
                action=ActionConfig(
                    type=str(self.env.action.type),
                ),
                obs=ObservationConfig(
                    kind=str(self.env.obs.kind),
                    view=str(self.env.obs.view),
                    params=dict(self.env.obs.params),
                    features=dict(self.env.obs.features),
                ),
                frame_stack=FrameStackConfig(
                    n_frames=int(self.env.frame_stack.n_frames),
                ),
            ),
            feature_extractor=FeaturesExtractorConfig(
                type=str(self.feature_extractor.type),
                features_dim=int(self.feature_extractor.features_dim),
                params=dict(self.feature_extractor.params),
            ),
            train=TrainLoopConfig(
                algo=AlgoConfig(
                    type=str(self.train.algo.type),
                    params=dict(self.train.algo.params),
                ),
                eval=EvalConfig(
                    enabled=bool(self.train.eval.enabled),
                    episodes=int(self.train.eval.episodes),
                    best_metric=str(self.train.eval.best_metric),
                    deterministic=bool(self.train.eval.deterministic),
                    seed_offset=int(self.train.eval.seed_offset),
                ),
            ),
        )


__all__ = [
    "AlgoConfigModel",
    "ActionConfigModel",
    "EnvConfigModel",
    "EvalConfigModel",
    "FeaturesExtractorConfigModel",
    "FrameStackConfigModel",
    "BoardConfigModel",
    "ObservationConfigModel",
    "RunConfigModel",
    "RewardConfigModel",
    "TrainLoopConfigModel",
    "TrainConfigModel",
]
