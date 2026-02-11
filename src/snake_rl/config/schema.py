# src/snake_rl/config/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
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
class RewardConfig:
    """
    Reward shaping configuration for Snake environments.

    Values are interpreted as:
      - win_reward: added on win
      - food_reward: added per food eaten
      - food_speed_bonus: added scaled by speed-to-food (1.0 - steps_since_food/max_steps)
      - fatal_penalty: subtracted on death
      - step_penalty_scale: scaled by 1/max_steps and subtracted each step
      - timeout_penalty: subtracted on timeout truncation
      - max_steps_factor: multiplier for max_steps = max_playable_tiles * factor
    """

    max_steps_factor: float = 1.3
    win_reward: float = 5.0
    food_reward: float = 2.5
    food_speed_bonus: float = 1.0
    fatal_penalty: float = 5.0
    step_penalty_scale: float = 1.0
    timeout_penalty: float = 0.0


@dataclass(frozen=True)
class EnvConfig:
    id: str
    # Parameters passed to the selected env constructor (e.g. view_radius for POV envs).
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameStackConfig:
    """
    Frame stacking configuration.

    n_frames = 1 means no stacking.
    """

    n_frames: int = 1


@dataclass(frozen=True)
class ObservationConfig:
    # Back-compat: older configs used observation.params for env kwargs.
    # Going forward, env kwargs belong in env.params.
    params: dict[str, Any] = field(default_factory=dict)
    frame_stack: FrameStackConfig = field(default_factory=FrameStackConfig)


# ---------------------------------------------------------------------------
# Model configuration (generic feature extractor, CNN or Transformer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeaturesExtractorConfig:
    """
    Feature extractor ("model") configuration.

    In snake_rl, models are SB3 feature extractors that map observations to
    feature vectors. The PPO policy head (action/value networks) is kept fixed.

    type:
      Feature extractor key (see models/registry.py), e.g.:
        - px_*    : pixel-based CNNs
        - tile_*  : symbolic tile-id models (MLP, ViT, ...)

    features_dim:
      Output feature dimension exposed to the policy MLP.

    params:
      Free-form extractor-specific parameters (passed through to the
      feature extractor constructor).
    """

    type: str
    features_dim: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    features_extractor: FeaturesExtractorConfig
    net_arch: list[int]


# ---------------------------------------------------------------------------
# PPO configuration (pass-through SB3 kwargs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalPhaseConfig:
    enabled: bool = False
    episodes: int = 10
    best_metric: str = "mean_reward"
    deterministic: bool = True
    seed_offset: int = 10_000


@dataclass(frozen=True)
class EvalConfig:
    intermediate: EvalPhaseConfig = field(default_factory=EvalPhaseConfig)
    final: EvalPhaseConfig = field(
        default_factory=lambda: EvalPhaseConfig(enabled=True, episodes=100, seed_offset=20_000)
    )


# ---------------------------------------------------------------------------
# Training configuration (algorithm + eval)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainLoopConfig:
    algo: str = "ppo"
    algo_params: dict[str, Any] = field(default_factory=dict)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ---------------------------------------------------------------------------
# Top-level training configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    run: RunConfig
    level: LevelConfig
    env: EnvConfig
    observation: ObservationConfig
    model: ModelConfig
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
