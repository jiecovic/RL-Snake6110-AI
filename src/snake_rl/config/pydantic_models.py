# src/snake_rl/config/pydantic_models.py

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from snake_rl.config.schema import (
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


class EnvConfigModel(_BaseConfigModel):
    id: str
    params: dict[str, Any] = Field(default_factory=dict)


class FrameStackConfigModel(_BaseConfigModel):
    n_frames: int = 1

    @field_validator("n_frames")
    @classmethod
    def _n_frames_at_least_one(cls, v: int) -> int:
        if int(v) <= 0:
            raise ValueError("observation.frame_stack.n_frames must be >= 1")
        return int(v)


class ObservationConfigModel(_BaseConfigModel):
    params: dict[str, Any] = Field(default_factory=dict)
    frame_stack: FrameStackConfigModel = Field(default_factory=FrameStackConfigModel)


class FeaturesExtractorConfigModel(_BaseConfigModel):
    type: str
    features_dim: int
    params: dict[str, Any] = Field(default_factory=dict)

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

    @model_validator(mode="before")
    @classmethod
    def _normalize_train_algo(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        out: dict[str, Any] = dict(data)
        algo_val = out.get("algo")
        eval_val = out.get("eval")

        # Back-compat: eval with {intermediate, final} -> pick final, else intermediate.
        if isinstance(eval_val, dict) and ("intermediate" in eval_val or "final" in eval_val):
            if "final" in eval_val and isinstance(eval_val.get("final"), dict):
                out["eval"] = eval_val.get("final")
            elif "intermediate" in eval_val and isinstance(eval_val.get("intermediate"), dict):
                out["eval"] = eval_val.get("intermediate")

        # Back-compat: algo: "ppo" + algo_params: {...}
        if "algo_params" in out:
            raw_params = out.pop("algo_params")
            params = raw_params if isinstance(raw_params, dict) else {}
            if isinstance(algo_val, dict):
                algo_dict: dict[str, Any] = dict(algo_val)
            elif isinstance(algo_val, str):
                algo_dict = {"type": algo_val}
            else:
                algo_dict = {}
            existing_params = algo_dict.get("params")
            merged = dict(existing_params) if isinstance(existing_params, dict) else {}
            merged.update(params)
            algo_dict["params"] = merged
            out["algo"] = algo_dict
            return out

        # Back-compat: algo: "ppo" with flat params at train level
        if isinstance(algo_val, str):
            extra = {k: v for k, v in out.items() if k not in {"algo", "eval"}}
            if extra:
                out = {k: v for k, v in out.items() if k in {"algo", "eval"}}
                out["algo"] = {"type": algo_val, "params": extra}
        return out


class TrainConfigModel(_BaseConfigModel):
    run: RunConfigModel
    board: BoardConfigModel
    reward: RewardConfigModel = Field(default_factory=RewardConfigModel)
    env: EnvConfigModel
    observation: ObservationConfigModel
    feature_extractor: FeaturesExtractorConfigModel
    train: TrainLoopConfigModel = Field(default_factory=TrainLoopConfigModel)

    @model_validator(mode="before")
    @classmethod
    def _map_level_to_board(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "board" in data or "level" not in data:
            return data
        out = dict(data)
        out["board"] = out.pop("level")
        return out

    @model_validator(mode="before")
    @classmethod
    def _map_model_to_feature_extractor(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        if "feature_extractor" in data or "model" not in data:
            return data

        model = data.get("model")
        if not isinstance(model, dict):
            return data

        out: dict[str, Any] = dict(data)
        out.pop("model", None)

        fe = model.get("features_extractor")
        if fe is None:
            fe = {k: v for k, v in model.items() if k != "net_arch"}
        out["feature_extractor"] = fe

        net_arch = model.get("net_arch")
        if net_arch is not None:
            train = out.get("train")
            train_dict: dict[str, Any] = dict(train) if isinstance(train, dict) else {}

            algo_val = train_dict.get("algo")
            if isinstance(algo_val, dict):
                algo_dict: dict[str, Any] = dict(algo_val)
            elif isinstance(algo_val, str):
                algo_dict = {"type": algo_val}
            else:
                algo_dict = {}

            params = algo_dict.get("params")
            params_dict = dict(params) if isinstance(params, dict) else {}
            policy_kwargs = params_dict.get("policy_kwargs")
            policy_dict = dict(policy_kwargs) if isinstance(policy_kwargs, dict) else {}
            policy_dict.setdefault("net_arch", net_arch)
            params_dict["policy_kwargs"] = policy_dict
            algo_dict["params"] = params_dict
            train_dict["algo"] = algo_dict
            out["train"] = train_dict

        return out

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
                id=str(self.env.id),
                params=dict(self.env.params),
            ),
            observation=ObservationConfig(
                params=dict(self.observation.params),
                frame_stack=FrameStackConfig(
                    n_frames=int(self.observation.frame_stack.n_frames),
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
