# src/snake_rl/config/pydantic_models.py

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from snake_rl.config.schema import (
    ActionConfig,
    AlgoConfig,
    BoardConfig,
    CheckpointConfig,
    EnvConfig,
    EvalConfig,
    FeaturesExtractorConfig,
    FrameStackConfig,
    MetricsConfig,
    MetricsGroupConfig,
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
    num_envs: int = 12
    vec: str = "dummy"
    checkpoint: "CheckpointConfigModel" = Field(default_factory=lambda: CheckpointConfigModel())
    total_timesteps: int
    resume_checkpoint: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_checkpoint_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        checkpoint = dict(payload.get("checkpoint") or {})
        if "checkpoint_freq" in payload and "freq" not in checkpoint:
            checkpoint["freq"] = payload.pop("checkpoint_freq")
        if "eval" in payload and "eval" not in checkpoint:
            checkpoint["eval"] = payload.pop("eval")
        if checkpoint:
            payload["checkpoint"] = checkpoint
        return payload

    @field_validator("vec")
    @classmethod
    def _normalize_vec(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s == "auto":
            return "dummy"
        if s not in {"dummy", "subproc", "rust"}:
            raise ValueError("run.vec must be one of: dummy, subproc, rust")
        return s


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
    action: ActionConfigModel = Field(default_factory=ActionConfigModel)
    obs: ObservationConfigModel
    frame_stack: FrameStackConfigModel = Field(default_factory=FrameStackConfigModel)

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_engine(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.pop("engine", None)
        data.pop("num_envs", None)
        data.pop("vec", None)
        return data


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
    deterministic: bool = True
    seed_offset: int = 10_000

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_best_metric(cls, data: Any) -> Any:
        if isinstance(data, dict) and "best_metric" in data:
            data = dict(data)
            data.pop("best_metric", None)
        return data


class CheckpointConfigModel(_BaseConfigModel):
    freq: int = 10_000
    eval: EvalConfigModel = Field(default_factory=EvalConfigModel)


class MetricsGroupConfigModel(_BaseConfigModel):
    keys: list[str] = Field(default_factory=list)
    termination: bool = True


class MetricsConfigModel(_BaseConfigModel):
    eval: MetricsGroupConfigModel = Field(default_factory=MetricsGroupConfigModel)
    train: MetricsGroupConfigModel = Field(default_factory=MetricsGroupConfigModel)


class TrainLoopConfigModel(_BaseConfigModel):
    algo: AlgoConfigModel = Field(default_factory=AlgoConfigModel)


class TrainConfigModel(_BaseConfigModel):
    run: RunConfigModel
    board: BoardConfigModel
    reward: RewardConfigModel = Field(default_factory=RewardConfigModel)
    env: EnvConfigModel
    feature_extractor: FeaturesExtractorConfigModel
    train: TrainLoopConfigModel = Field(default_factory=TrainLoopConfigModel)
    metrics: MetricsConfigModel = Field(default_factory=MetricsConfigModel)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_algo_group(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if "algo" in payload and "train" not in payload:
            train = {"algo": payload.pop("algo")}
            payload["train"] = train
        elif "train" in payload and "algo" in payload:
            payload.pop("algo", None)
        return payload

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_env_vec(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        run = dict(payload.get("run") or {})
        env = dict(payload.get("env") or {})
        if "num_envs" not in run and "num_envs" in env:
            run["num_envs"] = env.pop("num_envs")
        if "vec" not in run and "vec" in env:
            run["vec"] = env.pop("vec")
        if env:
            payload["env"] = env
        if run:
            payload["run"] = run
        return payload

    @model_validator(mode="before")
    @classmethod
    def _lift_eval_to_run(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        run = dict(payload.get("run") or {})
        train = dict(payload.get("train") or {})
        checkpoint = dict(run.get("checkpoint") or {})

        if "eval" in payload and "eval" not in run:
            checkpoint["eval"] = payload.pop("eval")

        if "eval" in train and "eval" not in run:
            checkpoint["eval"] = train.pop("eval")
            payload["train"] = train

        if "checkpoint_freq" in payload and "checkpoint" not in run:
            checkpoint["freq"] = payload.pop("checkpoint_freq")

        if "checkpoint_freq" in run and "freq" not in checkpoint:
            checkpoint["freq"] = run.pop("checkpoint_freq")

        if checkpoint:
            run["checkpoint"] = checkpoint

        if run:
            payload["run"] = run
        return payload

    def to_dataclass(self) -> TrainConfig:
        return TrainConfig(
            run=RunConfig(
                name=self.run.name,
                seed=int(self.run.seed),
                num_envs=int(self.run.num_envs),
                vec=str(self.run.vec),
                checkpoint=CheckpointConfig(
                    freq=int(self.run.checkpoint.freq),
                    eval=EvalConfig(
                        enabled=bool(self.run.checkpoint.eval.enabled),
                        episodes=int(self.run.checkpoint.eval.episodes),
                        deterministic=bool(self.run.checkpoint.eval.deterministic),
                        seed_offset=int(self.run.checkpoint.eval.seed_offset),
                    ),
                ),
                total_timesteps=int(self.run.total_timesteps),
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
            ),
            metrics=MetricsConfig(
                eval=MetricsGroupConfig(
                    keys=list(self.metrics.eval.keys),
                    termination=bool(self.metrics.eval.termination),
                ),
                train=MetricsGroupConfig(
                    keys=list(self.metrics.train.keys),
                    termination=bool(self.metrics.train.termination),
                ),
            ),
        )


__all__ = [
    "AlgoConfigModel",
    "ActionConfigModel",
    "CheckpointConfigModel",
    "EnvConfigModel",
    "EvalConfigModel",
    "MetricsConfigModel",
    "MetricsGroupConfigModel",
    "FeaturesExtractorConfigModel",
    "FrameStackConfigModel",
    "BoardConfigModel",
    "ObservationConfigModel",
    "RunConfigModel",
    "RewardConfigModel",
    "TrainLoopConfigModel",
    "TrainConfigModel",
]
