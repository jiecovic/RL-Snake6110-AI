# src/snake_rl/utils/model_params.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class ParamCount:
    total: int
    trainable: int


def _count_params(module: torch.nn.Module) -> ParamCount:
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return ParamCount(total=total, trainable=trainable)


def _get_attr(obj: Any, name: str) -> Optional[Any]:
    return getattr(obj, name, None)


def _format_count(n: int) -> str:
    # 20 -> "20", 1200 -> "1.2K", 2_340_000 -> "2.34M", 1_200_000_000 -> "1.20B"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_sb3_param_summary(model: Any) -> str:
    """
    One-line, human readable summary.
    Example: "[params] total=3.42M (trainable=3.42M)"
    """
    policy = _get_attr(model, "policy")
    if policy is None or not isinstance(policy, torch.nn.Module):
        return "[params] total=? (trainable=?)"

    pc = _count_params(policy)
    return f"[params] total={_format_count(pc.total)} (trainable={_format_count(pc.trainable)})"


def format_sb3_param_report(model: Any) -> str:
    """
    Works for SB3 algorithms that have `.policy` (e.g., PPO/A2C/SAC/TD3).
    Tries to be robust across CNN/ViT/MLP feature extractors.
    """
    policy = _get_attr(model, "policy")
    if policy is None:
        return "[params] model has no .policy attribute"

    lines: list[str] = []

    pc_policy = _count_params(policy)
    lines.append(f"[params] policy: total={pc_policy.total:,} trainable={pc_policy.trainable:,}")

    # Feature extractor (CNN / ViT / custom)
    feat = _get_attr(policy, "features_extractor")
    if isinstance(feat, torch.nn.Module):
        pc_feat = _count_params(feat)
        feat_name = type(feat).__name__
        lines.append(
            f"[params] features_extractor({feat_name}): total={pc_feat.total:,} trainable={pc_feat.trainable:,}"
        )

    # SB3 common: mlp_extractor with policy_net/value_net inside
    mlp_extractor = _get_attr(policy, "mlp_extractor")
    if isinstance(mlp_extractor, torch.nn.Module):
        pc_mlp = _count_params(mlp_extractor)
        lines.append(
            f"[params] mlp_extractor({type(mlp_extractor).__name__}): total={pc_mlp.total:,} trainable={pc_mlp.trainable:,}"
        )

        policy_net = _get_attr(mlp_extractor, "policy_net")
        if isinstance(policy_net, torch.nn.Module):
            pc = _count_params(policy_net)
            lines.append(f"[params]  ├─ policy_net: total={pc.total:,} trainable={pc.trainable:,}")

        value_net = _get_attr(mlp_extractor, "value_net")
        if isinstance(value_net, torch.nn.Module):
            pc = _count_params(value_net)
            lines.append(f"[params]  └─ value_net : total={pc.total:,} trainable={pc.trainable:,}")

    # Actor/Critic direct modules (some policies expose these)
    for name in ("action_net", "value_net", "actor", "critic", "qf0", "qf1"):
        m = _get_attr(policy, name)
        if isinstance(m, torch.nn.Module):
            pc = _count_params(m)
            lines.append(f"[params] {name}({type(m).__name__}): total={pc.total:,} trainable={pc.trainable:,}")

    return "\n".join(lines)
