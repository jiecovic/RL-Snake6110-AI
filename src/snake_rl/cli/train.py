# src/snake_rl/cli/train.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from snake_rl.config.pydantic_models import TrainConfigModel
from snake_rl.training.train_loop import train

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _pop_logging(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = data.pop("logging", {})
    if isinstance(raw, dict):
        return raw
    return {}


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    raw_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected hydra config to resolve to dict, got {type(raw).__name__}")

    raw = cast(Dict[str, Any], raw)
    raw.pop("hydra", None)
    logging_cfg = _pop_logging(raw)

    model = TrainConfigModel.model_validate(raw)
    train_cfg = model.to_dataclass()

    no_rich = bool(logging_cfg.get("no_rich", False))
    log_level = str(logging_cfg.get("level", "INFO"))
    train(
        cfg=train_cfg,
        use_rich=not no_rich,
        log_level=log_level,
        config_hydra_yaml=raw_yaml,
        config_validated=model.model_dump(mode="python"),
    )


if __name__ == "__main__":
    main()  # type: ignore[call-arg]
