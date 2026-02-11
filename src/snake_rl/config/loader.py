# src\snake_rl\config\loader.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from snake_rl.config.pydantic_models import TrainConfigModel
from snake_rl.config.schema import TrainConfig

RawConfig = dict[str, Any]


def _strip_non_train_keys(raw: RawConfig) -> tuple[RawConfig, dict[str, Any]]:
    data = dict(raw)
    data.pop("hydra", None)
    logging_cfg = data.pop("logging", {})
    if not isinstance(logging_cfg, dict):
        logging_cfg = {}
    return data, dict(logging_cfg)


def model_from_raw(raw: RawConfig) -> tuple[TrainConfigModel, dict[str, Any]]:
    payload, logging_cfg = _strip_non_train_keys(raw)
    model = TrainConfigModel.model_validate(payload)
    return model, logging_cfg


def dataclass_from_raw(raw: RawConfig) -> tuple[TrainConfig, dict[str, Any], TrainConfigModel]:
    model, logging_cfg = model_from_raw(raw)
    return model.to_dataclass(), logging_cfg, model


def load_from_hydra_cfg(cfg: DictConfig) -> tuple[RawConfig, str]:
    raw_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected hydra config to resolve to dict, got {type(raw).__name__}")
    return {str(k): v for k, v in raw.items()}, raw_yaml


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def compose_from_path(
    *,
    config_path: Path,
    config_root: Path,
    overrides: list[str] | None = None,
) -> DictConfig:
    rel = config_path.resolve().relative_to(config_root.resolve())
    config_name = str(rel.with_suffix(""))
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_root)):
        return hydra.compose(config_name=config_name, overrides=list(overrides or []))


def load_raw_from_path(
    *,
    config_path: Path,
    config_root: Path,
    overrides: list[str] | None = None,
) -> tuple[RawConfig, str | None]:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    if _is_under(config_path, config_root):
        cfg = compose_from_path(
            config_path=config_path,
            config_root=config_root,
            overrides=overrides,
        )
        raw, raw_yaml = load_from_hydra_cfg(cfg)
        return raw, raw_yaml

    text = config_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"{config_path.name} must parse to a dict, got {type(data).__name__}")
    return dict(data), text


def load_train_config_from_path(
    *,
    config_path: Path,
    config_root: Path,
    overrides: list[str] | None = None,
) -> tuple[TrainConfig, dict[str, Any], TrainConfigModel, str | None]:
    raw, raw_yaml = load_raw_from_path(
        config_path=config_path,
        config_root=config_root,
        overrides=overrides,
    )
    model, logging_cfg = model_from_raw(raw)
    return model.to_dataclass(), logging_cfg, model, raw_yaml


__all__ = [
    "RawConfig",
    "compose_from_path",
    "dataclass_from_raw",
    "load_from_hydra_cfg",
    "load_raw_from_path",
    "load_train_config_from_path",
    "model_from_raw",
]
