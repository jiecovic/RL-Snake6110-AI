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
    def _normalize_env_obs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        out = dict(data)
        env = out.get("env")
        env_d = dict(env) if isinstance(env, dict) else {}

        obs_top = out.get("observation")
        obs_from_env = env_d.get("obs", env_d.get("observation"))
        obs_d = dict(obs_from_env) if isinstance(obs_from_env, dict) else {}

        obs_params = dict(obs_d.get("params", {})) if isinstance(obs_d.get("params"), dict) else {}
        obs_features = (
            dict(obs_d.get("features", {})) if isinstance(obs_d.get("features"), dict) else {}
        )

        # Legacy: env.params -> obs.params (with optional env.engine)
        env_params = env_d.get("params")
        env_params_d = dict(env_params) if isinstance(env_params, dict) else {}
        if "engine" in env_params_d and "engine" not in env_d:
            env_d["engine"] = env_params_d.pop("engine")

        # Legacy: top-level observation.params -> obs.params
        top_params: dict[str, Any] = {}
        if isinstance(obs_top, dict):
            params_top = obs_top.get("params")
            if isinstance(params_top, dict):
                top_params = dict(params_top)

            frame_stack = obs_top.get("frame_stack")
            if isinstance(frame_stack, dict) and "frame_stack" not in env_d:
                env_d["frame_stack"] = dict(frame_stack)

        merged_params: dict[str, Any] = {}
        merged_params.update(top_params)
        merged_params.update(env_params_d)
        merged_params.update(obs_params)

        # Legacy: env.id -> obs.kind/view (+ features)
        env_id = env_d.get("id")
        if isinstance(env_id, str) and (("kind" not in obs_d) or ("view" not in obs_d)):
            key = env_id.strip().lower()
            mapping = {
                "world_pixel": ("pixel", "world", {}),
                "world_pixel_dir": ("pixel", "world", {"direction": True}),
                "head_pixel": ("pixel", "head", {}),
                "head_pixel_fill": ("pixel", "head", {"fill": {"enabled": True}}),
                "world_tile_id": ("tile_id", "world", {}),
                "head_tile_id": ("tile_id", "head", {}),
            }
            if key in mapping:
                kind, view, feats = mapping[key]
                obs_d.setdefault("kind", kind)
                obs_d.setdefault("view", view)
                obs_features = {**feats, **obs_features}

        # Legacy: fill_bins -> features.fill.bins
        if "fill_bins" in merged_params:
            fill_bins = merged_params.pop("fill_bins")
            f = obs_features.get("fill")
            if isinstance(f, dict):
                f2 = dict(f)
                f2.setdefault("enabled", True)
                f2.setdefault("bins", fill_bins)
                obs_features["fill"] = f2
            elif f is True or f is None:
                obs_features["fill"] = {"enabled": True, "bins": fill_bins}

        if merged_params:
            obs_d["params"] = merged_params
        if obs_features:
            obs_d["features"] = obs_features

        if obs_d:
            env_d["obs"] = obs_d

        # Drop legacy keys
        env_d.pop("id", None)
        env_d.pop("params", None)
        env_d.pop("observation", None)
        out.pop("observation", None)

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
