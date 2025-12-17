# src/snake_rl/config/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    num_envs: int
    total_timesteps: int
    checkpoint_freq: int
    resume_checkpoint: Optional[str] = None


@dataclass(frozen=True)
class EnvConfig:
    height: int
    width: int
    food_count: int


@dataclass(frozen=True)
class ObservationConfig:
    render_mode: Literal["none", "human", "rgb_array"]
    n_stack: int
    view_radius: int


@dataclass(frozen=True)
class CNNConfig:
    type: str
    output_dim: int


@dataclass(frozen=True)
class FeaturesExtractorConfig:
    type: str
    cnn: CNNConfig


@dataclass(frozen=True)
class ModelConfig:
    features_extractor: FeaturesExtractorConfig
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
class TrainConfig:
    run: RunConfig
    env: EnvConfig
    observation: ObservationConfig
    model: ModelConfig
    ppo: PPOConfig
