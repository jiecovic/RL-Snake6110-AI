# src/snake_rl/config/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoardConfig:
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
class ActionConfig:
    """
    Action specification for the environment.

    type:
      - relative: forward/left/right (0/1/2)
      - cardinal: up/right/down/left (0/1/2/3)
    """

    type: str = "relative"


@dataclass(frozen=True)
class FrameStackConfig:
    """
    Frame stacking configuration.

    n_frames = 1 means no stacking.
    """

    n_frames: int = 1


@dataclass(frozen=True)
class ObservationConfig:
    """
    Observation specification.

    kind:
      - pixel
      - categorical (uint8 tile ids; 0 is reserved for OOB)

    view:
      - world
      - head

    params:
      View/mode specific parameters (e.g. view_radius, rotate_to_head, remove_border).

    features:
      Deprecated. Features are now enabled via feature_extractor.params.feature_tokens.
    """

    kind: str
    view: str
    params: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvConfig:
    obs: ObservationConfig
    action: ActionConfig = field(default_factory=ActionConfig)
    frame_stack: FrameStackConfig = field(default_factory=FrameStackConfig)


# ---------------------------------------------------------------------------
# Feature extractor configuration (generic, CNN or Transformer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeaturesExtractorConfig:
    """
    Feature extractor configuration.

    In snake_rl, feature extractors map observations to feature vectors.
    The PPO policy head (action/value networks) is kept fixed.

    type:
      Feature extractor key (see models/registry.py).
      Use the unified extractor:
        - snake_unified: configurable stem + mixer + pooling

    features_dim:
      Output feature dimension exposed to the policy MLP.

    params:
      Free-form extractor-specific parameters (passed through to the
      feature extractor constructor).
    """

    type: str
    features_dim: int
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    enabled: bool = False
    episodes: int = 10
    deterministic: bool = True
    seed_offset: int = 10_000


@dataclass(frozen=True)
class CheckpointConfig:
    freq: int = 10_000
    eval: EvalConfig = field(default_factory=lambda: EvalConfig())


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    num_envs: int
    vec: str
    total_timesteps: int
    checkpoint: CheckpointConfig = field(default_factory=lambda: CheckpointConfig())
    resume_checkpoint: str | None = None


# ---------------------------------------------------------------------------
# Metrics logging configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsGroupConfig:
    keys: list[str] = field(default_factory=list)
    termination: bool = True


@dataclass(frozen=True)
class MetricsConfig:
    eval: MetricsGroupConfig = field(default_factory=MetricsGroupConfig)
    train: MetricsGroupConfig = field(default_factory=MetricsGroupConfig)


# ---------------------------------------------------------------------------
# Training algorithm configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlgoConfig:
    type: str = "ppo"
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Training configuration (algorithm + eval)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainLoopConfig:
    algo: AlgoConfig = field(default_factory=AlgoConfig)


# ---------------------------------------------------------------------------
# Top-level training configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    run: RunConfig
    board: BoardConfig
    env: EnvConfig
    feature_extractor: FeaturesExtractorConfig
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainLoopConfig = field(default_factory=TrainLoopConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
