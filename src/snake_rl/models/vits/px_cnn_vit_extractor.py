# src/snake_rl/models/vits/px_cnn_vit_extractor.py
from __future__ import annotations

from typing import Tuple

import importlib
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from snake_rl.models.cnns.base import BaseCNNExtractor


def _infer_chw(space: spaces.Box) -> Tuple[int, int, int]:
    if not isinstance(space, spaces.Box):
        raise TypeError(f"CnnViTExtractor expects spaces.Box, got {type(space)!r}")
    shape = space.shape
    if shape is None or len(shape) != 3:
        raise ValueError(f"CnnViTExtractor expects (C,H,W), got {shape!r}")
    c, h, w = int(shape[0]), int(shape[1]), int(shape[2])
    return c, h, w


def _available_stems_from_registry() -> list[str]:
    """
    List only CNN extractors from the global FEATURE_EXTRACTOR_REGISTRY.
    Done lazily to avoid circular imports.
    """
    mod = importlib.import_module("snake_rl.models.registry")
    reg = getattr(mod, "FEATURE_EXTRACTOR_REGISTRY")
    keys = []
    for k, cls in reg.items():
        try:
            if issubclass(cls, BaseCNNExtractor):
                keys.append(k)
        except Exception:
            continue
    return sorted(keys)


def _build_stem_from_key(*, key: str, observation_space: spaces.Box, c_mult: int = 1) -> nn.Module:
    """
    Resolve cnn_stem via FEATURE_EXTRACTOR_REGISTRY lazily (no circular imports),
    instantiate the CNN extractor, and take its stem.
    """
    k = str(key).strip().lower()

    mod = importlib.import_module("snake_rl.models.registry")
    reg = getattr(mod, "FEATURE_EXTRACTOR_REGISTRY")

    try:
        cls = reg[k]
    except KeyError as e:
        raise ValueError(f"Unknown cnn_stem={key!r}. Available CNN stems: {_available_stems_from_registry()}") from e

    if not isinstance(cls, type) or not issubclass(cls, BaseCNNExtractor):
        raise TypeError(
            f"cnn_stem={key!r} must refer to a BaseCNNExtractor in FEATURE_EXTRACTOR_REGISTRY; "
            f"got {cls!r}"
        )

    # Instantiate properly so any internal assumptions (self.in_ch, self.c_mult, etc.) are valid.
    # features_dim is irrelevant here; we only use forward_stem().
    cnn: BaseCNNExtractor = cls(
        observation_space,
        features_dim=1,
        normalized_image=False,
        c_mult=int(c_mult),
    )

    # Important: use the public contract
    return cnn.stem


class PxCnnViTExtractor(BaseFeaturesExtractor):
    """
    ViT-like encoder over CNN patch tokens for pixel observations.

    Pipeline:
      obs [B,C,H,W] -> cnn_stem -> featmap [B,C',H',W']
      -> 1x1 conv proj to d_model -> tokens [B, T=H'*W', d_model]
      -> +2D pos (+optional CLS) -> TransformerEncoder -> pool -> out_proj -> [B, features_dim]
    """

    def __init__(
            self,
            observation_space: spaces.Box,
            *,
            features_dim: int = 512,
            cnn_stem: str = "px_tilealign_cnn_c4",
            c_mult: int = 1,
            d_model: int = 128,
            n_layers: int = 4,
            n_heads: int = 4,
            ffn_dim: int | None = None,
            dropout: float = 0.1,
            use_cls_token: bool = True,
            use_2d_positional: bool = True,
            normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, int(features_dim))

        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"CnnViTExtractor expects Box, got {type(observation_space)!r}")

        if not is_image_space(observation_space, check_channels=False, normalized_image=bool(normalized_image)):
            raise ValueError(f"CnnViTExtractor requires an image Box space, got: {observation_space}")

        if int(c_mult) < 1:
            raise ValueError("c_mult must be >= 1")

        if int(d_model) <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}")
        if int(n_layers) <= 0:
            raise ValueError(f"n_layers must be > 0, got {n_layers}")
        if int(n_heads) <= 0 or (int(d_model) % int(n_heads) != 0):
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.normalized_image = bool(normalized_image)

        # Build CNN stem (spatial feature map, no Flatten)
        self.stem = _build_stem_from_key(
            key=str(cnn_stem),
            observation_space=observation_space,
            c_mult=int(c_mult),
        )

        # Probe stem output shape to define token grid
        with torch.no_grad():
            sample = observation_space.sample()
            if not isinstance(sample, np.ndarray):
                raise ValueError(f"Sampled observation was not a numpy array: {type(sample)}")
            x0 = torch.as_tensor(sample[None]).float()  # [1,C,H,W]
            y0 = self.stem(x0)
            if not isinstance(y0, torch.Tensor) or y0.ndim != 4:
                raise RuntimeError(
                    f"cnn_stem must return [B,C',H',W'], got {type(y0)} shape={getattr(y0, 'shape', None)}"
                )
            _, c_out, h_out, w_out = y0.shape

        self.cnn_out_channels = int(c_out)
        self.h = int(h_out)
        self.w = int(w_out)
        self.seq_len = int(self.h * self.w)

        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else 4 * int(d_model)
        self.dropout = float(dropout)
        self.use_cls_token = bool(use_cls_token)
        self.use_2d_positional = bool(use_2d_positional)

        # Project CNN channels -> transformer width
        self.in_proj = nn.Conv2d(self.cnn_out_channels, self.d_model, kernel_size=1, stride=1, padding=0, bias=True)

        # Positional embeddings on token grid (H', W')
        self.pos_1d = None
        self.pos_row = None
        self.pos_col = None
        if self.use_2d_positional:
            self.pos_row = nn.Embedding(self.h, self.d_model)
            self.pos_col = nn.Embedding(self.w, self.d_model)
        else:
            self.pos_1d = nn.Embedding(self.seq_len, self.d_model)

        self.cls_token = None
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.drop = nn.Dropout(self.dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.n_layers)

        self.out_proj = nn.Linear(self.d_model, int(features_dim), bias=True)

        # Init
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        if self.pos_row is not None:
            nn.init.normal_(self.pos_row.weight, mean=0.0, std=0.02)
        if self.pos_col is not None:
            nn.init.normal_(self.pos_col.weight, mean=0.0, std=0.02)
        if self.pos_1d is not None:
            nn.init.normal_(self.pos_1d.weight, mean=0.0, std=0.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _add_positional(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] where T = H'*W'
        if self.use_2d_positional:
            assert self.pos_row is not None and self.pos_col is not None
            device = x.device
            row_idx = torch.arange(self.h, device=device)
            col_idx = torch.arange(self.w, device=device)
            pe = (self.pos_row(row_idx)[:, None, :] + self.pos_col(col_idx)[None, :, :]).reshape(self.seq_len, self.d_model)
            return x + pe.unsqueeze(0)
        assert self.pos_1d is not None
        device = x.device
        idx = torch.arange(self.seq_len, device=device)
        pe = self.pos_1d(idx)
        return x + pe.unsqueeze(0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 4:
            raise ValueError(f"expected obs [B,C,H,W], got shape={tuple(observations.shape)}")

        x = observations.float()
        x = self.stem(x)  # [B, C', H', W']
        x = self.in_proj(x)  # [B, d_model, H', W']

        b, d, h, w = x.shape
        if h != self.h or w != self.w:
            raise ValueError(f"stem spatial changed: got {(h, w)} expected {(self.h, self.w)}")

        tokens = x.flatten(2).transpose(1, 2).contiguous()  # [B, T, d_model]
        tokens = self._add_positional(tokens)
        tokens = self.drop(tokens)

        if self.cls_token is not None:
            cls = self.cls_token.expand(b, 1, self.d_model)
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = self.encoder(tokens)

        pooled = tokens[:, 0, :] if self.cls_token is not None else tokens.mean(dim=1)
        return self.out_proj(pooled)
