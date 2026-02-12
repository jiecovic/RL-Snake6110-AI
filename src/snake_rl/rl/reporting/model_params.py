# src/snake_rl/rl/reporting/model_params.py
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from snake_rl.utils.model_params import format_sb3_param_report, format_sb3_param_summary
from snake_rl.utils.runs.paths import relpath


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _fmt_float(x: float) -> str:
    if x == 0.0:
        return "0"
    if 1e-3 <= abs(x) < 1e4:
        return f"{x:.6g}"
    return f"{x:.3e}"


def _try_relpath(value: Any, *, base: Path) -> str:
    if isinstance(value, Path):
        return relpath(value, base=base)
    if isinstance(value, str):
        try:
            return relpath(Path(value), base=base)
        except Exception:
            return value
    return str(value)


def _effective_ppo_init_kwargs(model: PPO) -> dict[str, Any]:
    sig = inspect.signature(PPO.__init__)
    keys = [k for k in sig.parameters if k != "self"]

    out: dict[str, Any] = {}
    for k in keys:
        if k in {"policy", "env"}:
            continue
        out[k] = getattr(model, k, "<not_exposed>")
    return out


def _log_block(logger, header: str, obj: Any) -> None:
    logger.info(header)
    if obj is None:
        logger.info("  <None>")
        return
    for line in repr(obj).splitlines():
        logger.info(line)


def log_policy_network_detailed(*, model: PPO, logger) -> None:
    policy = getattr(model, "policy", None)
    logger.info("Policy Network (detailed):")
    if policy is None:
        logger.info("  <no model.policy>")
        return

    logger.info(f"  policy_class: {policy.__class__.__name__}")

    feat = getattr(policy, "features_extractor", None)
    if feat is not None:
        _log_block(logger, "  features_extractor:", feat)

    mlp = getattr(policy, "mlp_extractor", None)
    if mlp is not None:
        _log_block(logger, "  mlp_extractor:", mlp)

    an = getattr(policy, "action_net", None)
    if an is not None:
        _log_block(logger, "  action_net:", an)

    vn = getattr(policy, "value_net", None)
    if vn is not None:
        _log_block(logger, "  value_net:", vn)


def log_ppo_params(*, model: PPO, cfg: Any, paths: Any, logger) -> None:
    """
    Log SB3 params and model structure.

    Accepts cfg/paths as Any so this can be reused by CLIs that load the snapshot YAML dict,
    without needing TrainConfig.
    """
    repo = getattr(paths, "repo_root", Path.cwd())

    logger.info("PPO effective params (SB3):")
    eff = _effective_ppo_init_kwargs(model)
    for k in sorted(eff.keys()):
        v = eff[k]
        if isinstance(v, float):
            vs = _fmt_float(v)
        elif isinstance(v, int):
            vs = _fmt_int(v)
        elif isinstance(v, (str, Path)) and (("log" in k) or ("path" in k) or k.endswith("_dir")):
            vs = _try_relpath(v, base=repo)
        else:
            vs = str(v)
        logger.info(f"  {k}: {vs}")

    logger.info("Model params (summary):")
    logger.info(format_sb3_param_summary(model))
    logger.info("Model params (detailed):")
    logger.info(format_sb3_param_report(model))

    log_policy_network_detailed(model=model, logger=logger)
