# src/snake_rl/models/registry.py
from __future__ import annotations

from typing import Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from snake_rl.models.cnns.px_tilealign_cnn_c4 import PxTileAlignedCNN_C4
from snake_rl.models.cnns.px_tilealign_linear_cnn_c8 import PxTileAlignLinearCNN_C8
from snake_rl.models.cnns.px_nature_cnn import PxNatureCNN
from snake_rl.models.cnns.px_lite_cnn_c8 import PxLiteCNN_C8
from snake_rl.models.vits.tile_vit_extractor import TileViTExtractor
from snake_rl.models.mlps.tile_mlp_extractor import TileMLPExtractor

FEATURE_EXTRACTOR_REGISTRY: dict[str, Type[BaseFeaturesExtractor]] = {
    # pixel-based CNNs
    "px_tilealign_cnn_c4": PxTileAlignedCNN_C4,
    "px_tilealign_linear_cnn_c8": PxTileAlignLinearCNN_C8,
    "px_nature_cnn": PxNatureCNN,
    "px_lite_cnn_c8": PxLiteCNN_C8,

    # symbolic tile-id models
    "tile_vit": TileViTExtractor,
    "tile_mlp": TileMLPExtractor,
}


def available_feature_extractors() -> list[str]:
    """Return sorted list of available feature extractor keys."""
    return sorted(FEATURE_EXTRACTOR_REGISTRY.keys())
