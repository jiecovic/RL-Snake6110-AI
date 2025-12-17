# src/snake_rl/models/cnns/registry.py
from typing import Type

from .base_cnn_extractor import BaseCNNExtractor
from .tile_cnn4 import TileCNN4
from .nature_cnn8 import NatureCNN8
from .lite_cnn8 import LiteCNN8

CNN_REGISTRY: dict[str, Type[BaseCNNExtractor]] = {
    "tile4": TileCNN4,
    "nature8": NatureCNN8,
    "lite8": LiteCNN8,
}
