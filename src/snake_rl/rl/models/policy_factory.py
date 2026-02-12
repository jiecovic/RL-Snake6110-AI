# src/snake_rl/rl/models/policy_factory.py
from __future__ import annotations

import inspect
from typing import Any

from torch import nn

from snake_rl.config.schema import TrainConfig
from snake_rl.models.registry import FEATURE_EXTRACTOR_REGISTRY, available_feature_extractors


def _filter_valid_extractor_kwargs(
    *,
    extractor_cls: type,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """
    Filter kwargs to only those accepted by extractor_cls.__init__.

    We keep configs strict:
      - Unknown keys => error (helps catch typos)
      - Known keys => passed through unchanged
    """
    sig = inspect.signature(extractor_cls.__init__)
    valid = set(sig.parameters.keys())
    valid.discard("self")

    unknown = sorted(k for k in kwargs if k not in valid)
    if unknown:
        raise ValueError(
            f"Extractor {extractor_cls.__name__} got unknown params: {unknown}. "
            f"Allowed: {sorted(valid)}"
        )

    return dict(kwargs)


def _extract_policy_kwargs_from_train(cfg: TrainConfig) -> dict[str, Any]:
    params = cfg.train.algo.params
    if not isinstance(params, dict):
        return {}
    policy_kwargs = params.get("policy_kwargs")
    if not isinstance(policy_kwargs, dict):
        return {}
    return dict(policy_kwargs)


def build_policy_kwargs(
    *,
    cfg: TrainConfig,
    observation_space,
    extra_policy_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build SB3 policy_kwargs from TrainConfig.

    In snake_rl, "models" are SB3 feature extractors. The PPO policy head remains SB3's
    default (ActorCritic*Policy), optionally with a post-extractor MLP (net_arch).

    Selection:
      cfg.feature_extractor.type must be a key in FEATURE_EXTRACTOR_REGISTRY.

    Convention:
      Use the unified extractor `snake_unified` with params describing the stem/mixer/pooling.
      See configs/feature_extractor/*.yaml for examples.
    """
    fe = cfg.feature_extractor
    extractor_key = str(fe.type).strip().lower()
    features_dim = int(fe.features_dim)
    extra_params = dict(fe.params)

    policy_extra = _extract_policy_kwargs_from_train(cfg)
    if extra_policy_kwargs:
        policy_extra.update(extra_policy_kwargs)

    net_arch = policy_extra.get("net_arch", [])

    try:
        extractor_cls = FEATURE_EXTRACTOR_REGISTRY[extractor_key]
    except KeyError as e:
        raise ValueError(
            f"Unknown feature extractor type {extractor_key!r}. "
            f"Available: {available_feature_extractors()}"
        ) from e

    policy_kwargs: dict[str, Any] = {
        "net_arch": list(net_arch) if isinstance(net_arch, list) else list(net_arch or []),
        "activation_fn": nn.GELU,  # oder nn.ReLU
        "features_extractor_class": extractor_cls,
    }

    for k, v in policy_extra.items():
        if k in {"features_extractor_class", "features_extractor_kwargs"}:
            continue
        policy_kwargs[k] = v

    extractor_kwargs = {
        "features_dim": int(features_dim),
        **extra_params,
    }
    extractor_kwargs = _filter_valid_extractor_kwargs(
        extractor_cls=extractor_cls,
        kwargs=extractor_kwargs,
    )
    policy_kwargs["features_extractor_kwargs"] = extractor_kwargs
    return policy_kwargs
