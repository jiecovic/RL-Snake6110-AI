# src/snake_rl/rl/reporting/model_params.py
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from snake_rl.utils.model_params import format_sb3_param_summary
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
        except (OSError, TypeError, ValueError):
            return value
    return str(value)


def _effective_ppo_init_kwargs(model: Any) -> dict[str, Any]:
    sig = inspect.signature(model.__class__.__init__)
    keys = [k for k in sig.parameters if k != "self"]

    out: dict[str, Any] = {}
    for k in keys:
        if k in {"policy", "env"}:
            continue
        out[k] = getattr(model, k, "<not_exposed>")
    return out


def _short(value: Any, *, base: Path) -> str:
    if isinstance(value, float):
        return _fmt_float(value)
    if isinstance(value, int):
        return _fmt_int(value)
    if isinstance(value, Path):
        return relpath(value, base=base)
    if isinstance(value, str):
        try:
            return relpath(Path(value), base=base)
        except (OSError, TypeError, ValueError):
            return value
    if isinstance(value, type):
        return value.__name__
    return str(value)


def _policy_summary(eff: dict[str, Any]) -> dict[str, Any]:
    pk = eff.get("policy_kwargs")
    if not isinstance(pk, dict):
        return {}
    out: dict[str, Any] = {}
    if "net_arch" in pk:
        out["net_arch"] = pk.get("net_arch")
    if "activation_fn" in pk:
        out["activation_fn"] = pk.get("activation_fn")
    if "features_extractor_class" in pk:
        out["features_extractor"] = pk.get("features_extractor_class")
    fx = pk.get("features_extractor_kwargs")
    if isinstance(fx, dict):
        out["features_dim"] = fx.get("features_dim")
        stem = fx.get("stem")
        if isinstance(stem, dict):
            out["stem"] = stem.get("type")
            params = stem.get("params")
            if isinstance(params, dict):
                out["stem_d_model"] = params.get("d_model")
                layers = params.get("layers")
                if isinstance(layers, list):
                    out["stem_layers"] = len(layers)
        mixer = fx.get("mixer")
        if isinstance(mixer, dict):
            out["mixer"] = mixer.get("type")
        out["pooling"] = fx.get("pooling")
        out["pos_mode"] = fx.get("pos_mode")
    return out


def log_model_layers(*, model: Any, logger, header: str = "[train] layers:") -> None:
    policy = getattr(model, "policy", None)
    logger.info(header)
    if policy is None:
        logger.info("  <no model.policy>")
        return
    logger.info(f"  policy={policy.__class__.__name__}")

    def _log(name: str) -> None:
        module = getattr(policy, name, None)
        if module is None:
            return
        logger.info(f"  {name}:")
        for line in repr(module).splitlines():
            logger.info(f"    {line}")

    _log("features_extractor")
    _log("mlp_extractor")
    _log("action_net")
    _log("value_net")


def log_ppo_params(*, model: Any, cfg: Any, paths: Any, logger, label: str = "train") -> None:
    """
    Log SB3 params and model structure.

    Accepts cfg/paths as Any so this can be reused by CLIs that load the snapshot YAML dict,
    without needing TrainConfig.
    """
    repo = getattr(paths, "repo_root", Path.cwd())

    eff = _effective_ppo_init_kwargs(model)
    label = str(label).strip() or "train"
    prefix = f"[{label}]"

    logger.info(f"{prefix} algo:")
    logger.info(
        "  rollout: "
        f"n_steps={_short(eff.get('n_steps'), base=repo)} "
        f"batch={_short(eff.get('batch_size'), base=repo)} "
        f"epochs={_short(eff.get('n_epochs'), base=repo)} "
        f"gamma={_short(eff.get('gamma'), base=repo)} "
        f"gae={_short(eff.get('gae_lambda'), base=repo)} "
        f"norm_adv={_short(eff.get('normalize_advantage'), base=repo)}"
    )
    logger.info(
        "  optim: "
        f"lr={_short(eff.get('learning_rate'), base=repo)} "
        f"clip={_short(eff.get('clip_range'), base=repo)} "
        f"vf={_short(eff.get('vf_coef'), base=repo)} "
        f"ent={_short(eff.get('ent_coef'), base=repo)} "
        f"max_grad={_short(eff.get('max_grad_norm'), base=repo)}"
    )
    logger.info(f"  device={_short(eff.get('device'), base=repo)}")

    policy = getattr(model, "policy", None)
    policy_name = policy.__class__.__name__ if policy is not None else "?"
    logger.info(f"{prefix} policy: {policy_name}")

    summary = _policy_summary(eff)
    if summary:
        parts = []
        for key in (
            "features_extractor",
            "features_dim",
            "stem",
            "stem_d_model",
            "stem_layers",
            "mixer",
            "pooling",
            "pos_mode",
            "net_arch",
            "activation_fn",
        ):
            if key in summary and summary[key] is not None:
                parts.append(f"{key}={_short(summary[key], base=repo)}")
        if parts:
            logger.info(f"{prefix} feature_extractor:")
            logger.info("  " + " ".join(parts))

    logger.info(f"{prefix} params:")
    logger.info(f"  {format_sb3_param_summary(model)}")

    log_model_layers(model=model, logger=logger, header=f"{prefix} layers:")
