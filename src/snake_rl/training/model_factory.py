# src/snake_rl/training/model_factory.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

from snake_rl.config.schema import TrainConfig
from snake_rl.models.cnns.registry import CNN_REGISTRY
from snake_rl.models.sb3.extractors import CustomCombinedExtractor


def build_policy_kwargs(*, cfg: TrainConfig, observation_space: spaces.Space) -> dict:
    cnn_key = cfg.model.cnn.type
    if cnn_key not in CNN_REGISTRY:
        raise KeyError(f"Unknown model.cnn.type={cnn_key!r}. Known: {sorted(CNN_REGISTRY.keys())}")

    cnn_cls = CNN_REGISTRY[cnn_key]
    features_dim = int(cfg.model.cnn.features_dim)

    if isinstance(observation_space, spaces.Dict):
        return {
            "net_arch": list(cfg.model.net_arch),
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "cnn_features_dim": features_dim,
                "cnn_extractor_class": cnn_cls,
            },
        }

    if isinstance(observation_space, spaces.Box):
        return {
            "net_arch": list(cfg.model.net_arch),
            "features_extractor_class": cnn_cls,
            "features_extractor_kwargs": {
                "features_dim": features_dim,
            },
        }

    raise TypeError(f"Unsupported observation space: {observation_space}")


def make_or_load_model(
    *,
    cfg: TrainConfig,
    vec_env,
    tensorboard_log: Path,
    resume_path: Optional[Path],
) -> PPO:
    if resume_path is not None:
        return PPO.load(str(resume_path), env=vec_env)

    policy_kwargs = build_policy_kwargs(cfg=cfg, observation_space=vec_env.observation_space)
    policy = MultiInputActorCriticPolicy if isinstance(vec_env.observation_space, spaces.Dict) else "CnnPolicy"

    return PPO(
        policy=policy,
        env=vec_env,
        n_steps=int(cfg.ppo.n_steps),
        batch_size=int(cfg.ppo.batch_size),
        n_epochs=int(cfg.ppo.n_epochs),
        gamma=float(cfg.ppo.gamma),
        ent_coef=float(cfg.ppo.ent_coef),
        learning_rate=float(cfg.ppo.learning_rate),
        verbose=int(cfg.ppo.verbose),
        seed=int(cfg.run.seed),
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log),
    )
