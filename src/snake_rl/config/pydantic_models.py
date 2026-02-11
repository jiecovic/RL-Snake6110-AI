from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from snake_rl.config.schema import (
    EnvConfig,
    EvalConfig,
    EvalPhaseConfig,
    FeaturesExtractorConfig,
    FrameStackConfig,
    LevelConfig,
    ModelConfig,
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
    resume_checkpoint: Optional[str] = None


class LevelConfigModel(_BaseConfigModel):
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


class EnvConfigModel(_BaseConfigModel):
    id: str
    params: Dict[str, Any] = Field(default_factory=dict)


class FrameStackConfigModel(_BaseConfigModel):
    n_frames: int = 1

    @field_validator("n_frames")
    @classmethod
    def _n_frames_at_least_one(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError("observation.frame_stack.n_frames must be >= 1")
        return int(v)


class ObservationConfigModel(_BaseConfigModel):
    params: Dict[str, Any] = Field(default_factory=dict)
    frame_stack: FrameStackConfigModel = Field(default_factory=FrameStackConfigModel)


class FeaturesExtractorConfigModel(_BaseConfigModel):
    type: str
    features_dim: int
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _back_compat_cnn_block(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "cnn" in data and "type" not in data:
            cnn_d = data.get("cnn")
            if not isinstance(cnn_d, dict):
                return data
            fe_type = cnn_d.get("type")
            fe_dim = cnn_d.get("features_dim")
            params = {str(k): v for k, v in cnn_d.items() if str(k) not in {"type", "features_dim"}}
            return {
                "type": fe_type,
                "features_dim": fe_dim,
                "params": params,
            }
        return data


class ModelConfigModel(_BaseConfigModel):
    features_extractor: FeaturesExtractorConfigModel
    net_arch: List[int]


class EvalPhaseConfigModel(_BaseConfigModel):
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
            raise ValueError("eval.*.best_metric must be one of: mean_reward, mean_score")
        return s


class EvalConfigModel(_BaseConfigModel):
    intermediate: EvalPhaseConfigModel = Field(default_factory=EvalPhaseConfigModel)
    final: EvalPhaseConfigModel = Field(
        default_factory=lambda: EvalPhaseConfigModel(
            enabled=True,
            episodes=100,
            best_metric="mean_reward",
            seed_offset=20_000,
        )
    )


class TrainLoopConfigModel(_BaseConfigModel):
    algo: str = "ppo"
    algo_params: Dict[str, Any] = Field(default_factory=dict)
    eval: EvalConfigModel = Field(default_factory=EvalConfigModel)

    @model_validator(mode="before")
    @classmethod
    def _wrap_flat_params(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "algo_params" not in data and ("algo" in data or "eval" in data):
            return data
        if "algo_params" in data:
            params = data.get("algo_params")
            if params is None:
                params = {}
            if not isinstance(params, dict):
                return data
            extra = {k: v for k, v in data.items() if k != "algo_params"}
            if not extra:
                return {"algo_params": dict(params)}
            merged = dict(params)
            merged.update(extra)
            return {"algo_params": merged}
        return {"algo_params": dict(data)}


class TrainConfigModel(_BaseConfigModel):
    run: RunConfigModel
    level: LevelConfigModel
    reward: RewardConfigModel = Field(default_factory=RewardConfigModel)
    env: EnvConfigModel
    observation: ObservationConfigModel
    model: ModelConfigModel
    train: TrainLoopConfigModel = Field(default_factory=TrainLoopConfigModel)

    @model_validator(mode="before")
    @classmethod
    def _merge_obs_params_into_env(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        obs = data.get("observation")
        if not isinstance(obs, dict):
            return data
        obs_params = obs.get("params")
        if not isinstance(obs_params, dict) or not obs_params:
            return data

        env = data.get("env")
        env_d = dict(env) if isinstance(env, dict) else {}
        env_params = env_d.get("params")
        if env_params is None:
            env_params = {}
        if not isinstance(env_params, dict):
            return data

        merged = dict(obs_params)
        merged.update(env_params)
        env_d["params"] = merged

        out = dict(data)
        out["env"] = env_d
        return out

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
            level=LevelConfig(
                height=int(self.level.height),
                width=int(self.level.width),
                food_count=int(self.level.food_count),
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
                id=str(self.env.id),
                params=dict(self.env.params),
            ),
            observation=ObservationConfig(
                params=dict(self.observation.params),
                frame_stack=FrameStackConfig(
                    n_frames=int(self.observation.frame_stack.n_frames),
                ),
            ),
            model=ModelConfig(
                features_extractor=FeaturesExtractorConfig(
                    type=str(self.model.features_extractor.type),
                    features_dim=int(self.model.features_extractor.features_dim),
                    params=dict(self.model.features_extractor.params),
                ),
                net_arch=list(self.model.net_arch),
            ),
            train=TrainLoopConfig(
                algo=str(self.train.algo),
                algo_params=dict(self.train.algo_params),
                eval=EvalConfig(
                    intermediate=EvalPhaseConfig(
                        enabled=bool(self.train.eval.intermediate.enabled),
                        episodes=int(self.train.eval.intermediate.episodes),
                        best_metric=str(self.train.eval.intermediate.best_metric),
                        deterministic=bool(self.train.eval.intermediate.deterministic),
                        seed_offset=int(self.train.eval.intermediate.seed_offset),
                    ),
                    final=EvalPhaseConfig(
                        enabled=bool(self.train.eval.final.enabled),
                        episodes=int(self.train.eval.final.episodes),
                        best_metric=str(self.train.eval.final.best_metric),
                        deterministic=bool(self.train.eval.final.deterministic),
                        seed_offset=int(self.train.eval.final.seed_offset),
                    ),
                ),
            ),
        )


__all__ = [
    "EnvConfigModel",
    "EvalConfigModel",
    "EvalPhaseConfigModel",
    "FeaturesExtractorConfigModel",
    "FrameStackConfigModel",
    "LevelConfigModel",
    "ModelConfigModel",
    "ObservationConfigModel",
    "RunConfigModel",
    "RewardConfigModel",
    "TrainLoopConfigModel",
    "TrainConfigModel",
]
