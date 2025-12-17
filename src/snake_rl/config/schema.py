# src/snake_rl/config/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    num_envs: int
    total_timesteps: int
    checkpoint_freq: int
    resume_checkpoint: Optional[str] = None


@dataclass(frozen=True)
class LevelConfig:
    height: int
    width: int
    food_count: int


@dataclass(frozen=True)
class EnvConfig:
    id: str


@dataclass(frozen=True)
class ObservationConfig:
    params: dict[str, Any]


@dataclass(frozen=True)
class CNNConfig:
    type: str
    features_dim: int


@dataclass(frozen=True)
class ModelConfig:
    cnn: CNNConfig
    net_arch: list[int]


@dataclass(frozen=True)
class PPOConfig:
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    ent_coef: float
    learning_rate: float
    verbose: int


@dataclass(frozen=True)
class EvalPhaseConfig:
    enabled: bool = False
    episodes: int = 10
    deterministic: bool = True
    seed_offset: int = 10_000


@dataclass(frozen=True)
class EvalConfig:
    intermediate: EvalPhaseConfig = EvalPhaseConfig()
    final: EvalPhaseConfig = EvalPhaseConfig(enabled=True, episodes=100, seed_offset=20_000)


@dataclass(frozen=True)
class TrainConfig:
    run: RunConfig
    level: LevelConfig
    env: EnvConfig
    observation: ObservationConfig
    model: ModelConfig
    ppo: PPOConfig
    eval: EvalConfig = EvalConfig()
