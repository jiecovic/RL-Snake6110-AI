# src/snake_rl/models/registry.py
from __future__ import annotations

from typing import Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# pixel-based CNNs
from snake_rl.models.cnns.px_nature_cnn import PxNatureCNN
from snake_rl.models.cnns.px_strided_cnn_l1k4 import PxStridedCNN_L1K4
from snake_rl.models.cnns.px_strided_cnn_l3k4 import PxStridedCNN_L3K4
from snake_rl.models.cnns.px_strided_cnn_l3k8 import PxStridedCNN_L3K8
from snake_rl.models.cnns.px_strided_cnn_l3k8_mask import PxStridedCNN_L3K8_Mask

# hybrid CNN → ViT
from snake_rl.models.vits.px_cnn_vit_extractor import PxCnnViTExtractor

# symbolic tile-id models
from snake_rl.models.mlps.tile_mlp_extractor import TileMLPExtractor
from snake_rl.models.vits.tile_vit_extractor import TileViTExtractor

FEATURE_EXTRACTOR_REGISTRY: dict[str, Type[BaseFeaturesExtractor]] = {
    # pixel-based CNNs
    "px_strided_cnn_l1k4": PxStridedCNN_L1K4,
    "px_strided_cnn_l3k4": PxStridedCNN_L3K4,
    "px_strided_cnn_l3k8": PxStridedCNN_L3K8,
    "px_strided_cnn_l3k8_mask": PxStridedCNN_L3K8_Mask,
    "px_nature_cnn": PxNatureCNN,
    # hybrid CNN → ViT
    "px_cnn_vit": PxCnnViTExtractor,
    # symbolic tile-id models
    "tile_vit": TileViTExtractor,
    "tile_mlp": TileMLPExtractor,
}


def available_feature_extractors() -> list[str]:
    """Return sorted list of available feature extractor keys."""
    return sorted(FEATURE_EXTRACTOR_REGISTRY.keys())
