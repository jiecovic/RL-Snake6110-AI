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
class PPOConfig:
    """
    Pass-through Stable-Baselines3 PPO kwargs.

    Users can add any SB3 PPO kwargs in YAML without changing Python code.
    Missing keys are fine: SB3 defaults apply.
    """
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalPhaseConfig:
    enabled: bool = False
    episodes: int = 10
    deterministic: bool = True
    seed_offset: int = 10_000


@dataclass(frozen=True)
class EvalConfig:
    intermediate: EvalPhaseConfig = field(default_factory=EvalPhaseConfig)
    final: EvalPhaseConfig = field(
        default_factory=lambda: EvalPhaseConfig(enabled=True, episodes=100, seed_offset=20_000)
    )


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
    ppo: PPOConfig
    eval: EvalConfig = field(default_factory=EvalConfig)
