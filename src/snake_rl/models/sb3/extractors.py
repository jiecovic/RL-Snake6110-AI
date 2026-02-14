# src/snake_rl/models/sb3/extractors.py
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from snake_rl.models.mixers import IdentityMixer, TransformerMixer
from snake_rl.models.stems import CatConvStem, CatIdentityStem, CatPatchStem, PxCnnStem
from snake_rl.models.stems.base import TokenStemOutput
from snake_rl.models.utils.feature_tokens import DEFAULT_CATEGORICAL, FeatureTokenBuilder
from snake_rl.models.utils.positional import GridPositionalEncoding, pool_tokens


def _infer_num_tiles(space: spaces.Box) -> int:
    hi = space.high
    max_hi = float(hi.max()) if hasattr(hi, "max") else float(hi)
    return int(max_hi) + 1


def _dummy_from_space(space: spaces.Box) -> torch.Tensor:
    sample = space.sample()
    if not isinstance(sample, np.ndarray):
        raise ValueError(f"Sampled observation was not a numpy array: {type(sample)!r}")
    return torch.as_tensor(sample[None])


def _normalize_hidden(value: int | list[int] | tuple[int, ...] | None) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return [int(v) for v in value]
    return [int(value)]


class SnakeExtractor(BaseFeaturesExtractor):
    """
    Unified SB3 features extractor.

    Pipeline: stem -> (pos/type enc) -> (feature tokens) -> mixer -> pool -> head
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        *,
        features_dim: int = 512,
        stem: dict[str, Any],
        mixer: dict[str, Any] | None = None,
        pooling: str = "mean",
        pos_mode: str = "abs_2d",
        use_cls_token: bool = True,
        use_frame_embed: bool = True,
        use_token_mask: bool = False,
        mask_token_id: int = 0,
        mask_pool: bool = True,
        feature_tokens: list[list[str]] | None = None,
        flatten_mlp_hidden: int | list[int] | None = None,
        flatten_mlp_hidden_dim: int | None = None,
        head_layernorm: bool = True,
        head_activation: str = "gelu",
        post_mlp_hidden: list[int] | None = None,
        post_mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__(observation_space, int(features_dim))
        if flatten_mlp_hidden is not None and flatten_mlp_hidden_dim is not None:
            raise ValueError("Use flatten_mlp_hidden or flatten_mlp_hidden_dim, not both.")
        if flatten_mlp_hidden is None and flatten_mlp_hidden_dim is not None:
            flatten_mlp_hidden = int(flatten_mlp_hidden_dim)

        stem_type = str(stem.get("type", "")).strip().lower()
        stem_params = dict(stem.get("params", {}))
        self.stem_kind = "pixel" if stem_type.startswith("px_") else "categorical"

        obs_space = self._select_obs_space(observation_space, self.stem_kind)
        self.obs_space = obs_space

        if self.stem_kind == "categorical":
            num_tiles = stem_params.pop("num_tiles", None)
            if num_tiles is None:
                num_tiles = _infer_num_tiles(obs_space)
            stem_params["num_tiles"] = int(num_tiles)

        self.stem = self._build_stem(
            stem_type=stem_type,
            obs_space=obs_space,
            stem_params=stem_params,
            use_token_mask=bool(use_token_mask),
            mask_token_id=int(mask_token_id),
        )

        dummy = _dummy_from_space(obs_space)
        if self.stem_kind == "categorical":
            dummy = dummy.to(dtype=torch.long)
        else:
            dummy = dummy.to(dtype=torch.float32)
        probe = self.stem(dummy)
        self.d_model = int(probe.tokens.shape[-1])
        self.grid = probe.grid
        self.frame_count = int(probe.frame_count)
        self.frame_fuse = str(probe.frame_fuse)
        self.tokens_per_frame = (
            int(self.grid[0] * self.grid[1]) if self.grid is not None else probe.tokens.shape[1]
        )

        self.use_frame_embed = bool(use_frame_embed) and self.frame_count > 1
        self.frame_emb: nn.Embedding | None = None
        if self.use_frame_embed:
            self.frame_emb = nn.Embedding(self.frame_count, self.d_model)
            nn.init.normal_(self.frame_emb.weight, mean=0.0, std=0.02)

        self.pos_enc: GridPositionalEncoding | None = None
        if self.grid is not None:
            pos_mode = str(pos_mode)
            if pos_mode and pos_mode != "none":
                self.pos_enc = GridPositionalEncoding(
                    h=int(self.grid[0]),
                    w=int(self.grid[1]),
                    d_model=self.d_model,
                    pos_mode=pos_mode,
                )

        self.use_cls_token = bool(use_cls_token)
        self.cls_token: torch.Tensor | None = None
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.feature_token_builder: FeatureTokenBuilder | None = None
        self.num_feature_tokens = 0
        feature_keys = self._feature_keys(observation_space)
        if feature_keys:
            if feature_tokens is None:
                cat = [k for k in feature_keys if k in DEFAULT_CATEGORICAL]
                num = [k for k in feature_keys if k not in DEFAULT_CATEGORICAL]
                groups: list[list[str]] = []
                groups.extend([[k] for k in cat])
                if num:
                    groups.append(list(num))
                feature_tokens = groups
            if feature_tokens:
                self.feature_token_builder = FeatureTokenBuilder(feature_tokens, self.d_model)
                self.num_feature_tokens = len(self.feature_token_builder.groups)

        self.mask_pool = bool(mask_pool)

        mixer_cfg = mixer or {"type": "identity"}
        mixer_type = str(mixer_cfg.get("type", "identity")).strip().lower()
        mixer_params = dict(mixer_cfg.get("params", {}))
        if mixer_type == "transformer":
            self.mixer = TransformerMixer(d_model=self.d_model, **mixer_params)
        elif mixer_type == "identity":
            self.mixer = IdentityMixer()
        else:
            raise ValueError(f"Unknown mixer type {mixer_type!r}")

        self.pooling = str(pooling)
        if self.pooling not in {"cls", "mean", "max", "cls_mean", "meanmax", "flatten"}:
            raise ValueError(f"Unsupported pooling {pooling!r}")
        if self.pooling in {"cls", "cls_mean"} and not self.use_cls_token:
            raise ValueError(f"pooling={self.pooling!r} requires use_cls_token=True")

        base_tokens = self.tokens_per_frame
        if self.frame_fuse == "concat" and self.frame_count > 1:
            base_tokens *= self.frame_count
        num_tokens = base_tokens + self.num_feature_tokens

        if self.pooling == "flatten":
            in_dim = num_tokens * self.d_model
        else:
            in_dim = self.d_model * (2 if self.pooling in {"cls_mean", "meanmax"} else 1)

        self.head = self._build_head(
            in_dim=in_dim,
            out_dim=int(features_dim),
            hidden=post_mlp_hidden,
            dropout=float(post_mlp_dropout),
            flatten_hidden=flatten_mlp_hidden,
            use_flatten=(self.pooling == "flatten"),
            use_layernorm=bool(head_layernorm),
            activation=str(head_activation),
        )

    def _select_obs_space(self, space: spaces.Space, kind: str) -> spaces.Box:
        if isinstance(space, spaces.Dict):
            key = "pixel" if kind == "pixel" else "categorical"
            if key not in space.spaces and key == "categorical" and "tiles" in space.spaces:
                key = "tiles"
            if key not in space.spaces:
                raise ValueError(f"Dict observation_space missing key {key!r}")
            sub = space.spaces[key]
            if not isinstance(sub, spaces.Box):
                raise ValueError(f"observation_space['{key}'] must be Box, got {type(sub)!r}")
            return sub
        if not isinstance(space, spaces.Box):
            raise ValueError(f"observation_space must be Box or Dict, got {type(space)!r}")
        return space

    def _feature_keys(self, space: spaces.Space) -> list[str]:
        if not isinstance(space, spaces.Dict):
            return []
        return [k for k in space.spaces if k not in {"pixel", "categorical", "tiles"}]

    def _build_stem(
        self,
        *,
        stem_type: str,
        obs_space: spaces.Box,
        stem_params: dict[str, Any],
        use_token_mask: bool,
        mask_token_id: int,
    ) -> nn.Module:
        in_channels = int(obs_space.shape[0]) if obs_space.shape is not None else 0
        frame_stack_n = getattr(obs_space, "_frame_stack_n", None)
        frame_base_channels = stem_params.get("frame_base_channels")
        if frame_base_channels is None:
            meta_base = getattr(obs_space, "_frame_base_channels", None)
            if meta_base is not None:
                frame_base_channels = int(meta_base)
            elif (
                frame_stack_n is not None
                and int(frame_stack_n) > 0
                and in_channels > 0
                and in_channels % int(frame_stack_n) == 0
            ):
                frame_base_channels = int(in_channels // int(frame_stack_n))
        if frame_base_channels is None:
            frame_base_channels = 1
        if stem_type == "px_cnn":
            return PxCnnStem(
                in_channels=in_channels,
                d_model=int(stem_params.get("d_model", 128)),
                layers=list(stem_params.get("layers", [])),
                frame_mode=str(stem_params.get("frame_mode", "stacked")),
                frame_base_channels=int(frame_base_channels),
            )
        if stem_type == "px_nature":
            return PxCnnStem(
                in_channels=in_channels,
                d_model=int(stem_params.get("d_model", 128)),
                preset="nature",
                frame_mode=str(stem_params.get("frame_mode", "stacked")),
                frame_base_channels=int(frame_base_channels),
            )
        if stem_type == "cat_identity":
            return CatIdentityStem(
                num_tiles=int(stem_params["num_tiles"]),
                d_model=int(stem_params.get("d_model", 128)),
                frame_fuse=str(stem_params.get("frame_fuse", "sum")),
                use_token_mask=bool(use_token_mask),
                mask_token_id=int(mask_token_id),
            )
        if stem_type == "cat_patch":
            return CatPatchStem(
                num_tiles=int(stem_params["num_tiles"]),
                d_model=int(stem_params.get("d_model", 128)),
                patch=int(stem_params.get("patch", 2)),
                stride=int(stem_params.get("stride", stem_params.get("patch", 2))),
                frame_fuse=str(stem_params.get("frame_fuse", "sum")),
                use_token_mask=bool(use_token_mask),
                mask_token_id=int(mask_token_id),
            )
        if stem_type == "cat_conv":
            return CatConvStem(
                num_tiles=int(stem_params["num_tiles"]),
                d_model=int(stem_params.get("d_model", 128)),
                layers=list(stem_params.get("layers", [])),
                frame_fuse=str(stem_params.get("frame_fuse", "sum")),
                use_token_mask=bool(use_token_mask),
                mask_token_id=int(mask_token_id),
            )
        raise ValueError(f"Unknown stem type {stem_type!r}")

    @staticmethod
    def _build_head(
        *,
        in_dim: int,
        out_dim: int,
        hidden: list[int] | None,
        dropout: float,
        flatten_hidden: int | list[int] | None,
        use_flatten: bool,
        use_layernorm: bool,
        activation: str,
    ) -> nn.Module:
        act_name = str(activation).strip().lower()
        if act_name in {"relu"}:
            act_layer = nn.ReLU
        elif act_name in {"gelu"}:
            act_layer = nn.GELU
        elif act_name in {"elu"}:
            act_layer = nn.ELU
        elif act_name in {"leaky_relu", "leakyrelu"}:
            act_layer = nn.LeakyReLU
        elif act_name in {"silu", "swish"}:
            act_layer = nn.SiLU
        elif act_name in {"tanh"}:
            act_layer = nn.Tanh
        else:
            raise ValueError(f"Unknown head_activation {activation!r}")

        if use_flatten:
            flat_hidden = _normalize_hidden(flatten_hidden)
            if flat_hidden is not None:
                layers: list[nn.Module] = []
                if use_layernorm:
                    layers.append(nn.LayerNorm(int(in_dim)))
                d = int(in_dim)
                for h in flat_hidden:
                    layers.append(nn.Linear(d, int(h)))
                    layers.append(act_layer())
                    d = int(h)
                if int(d) != int(out_dim):
                    layers.append(nn.Linear(int(d), int(out_dim)))
                return nn.Sequential(*layers)
        if not hidden:
            if int(in_dim) == int(out_dim):
                return nn.Identity()
            return nn.Linear(int(in_dim), int(out_dim))

        layers: list[nn.Module] = []
        d = int(in_dim)
        for h in hidden:
            layers.append(nn.Linear(d, int(h)))
            layers.append(act_layer())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            d = int(h)
        if int(d) != int(out_dim):
            layers.append(nn.Linear(d, int(out_dim)))
        return nn.Sequential(*layers)

    def _split_obs(
        self,
        observations: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(observations, dict):
            if self.stem_kind == "pixel":
                key = "pixel"
            else:
                key = "categorical" if "categorical" in observations else "tiles"
            if key not in observations:
                raise KeyError(f"observation dict missing key {key!r}")
            obs = observations[key]
            features = {
                k: v for k, v in observations.items() if k not in {"pixel", "categorical", "tiles"}
            }
            return obs, features
        return observations, {}

    def forward(self, observations: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        obs, features = self._split_obs(observations)
        out: TokenStemOutput = self.stem(obs)
        tokens = out.tokens
        mask = out.mask
        b = tokens.shape[0]

        if self.pos_enc is not None and out.grid is not None:
            if out.frame_fuse == "concat" and out.frame_count > 1:
                tpf = int(out.grid[0] * out.grid[1])
                tokens = tokens.view(b, out.frame_count, tpf, self.d_model)
                tokens = torch.cat(
                    [self.pos_enc(tokens[:, i, :, :]) for i in range(out.frame_count)],
                    dim=1,
                )
            else:
                tokens = self.pos_enc(tokens)

        if self.frame_emb is not None and out.frame_fuse == "concat" and out.frame_count > 1:
            tpf = int(tokens.shape[1] // out.frame_count)
            tokens = tokens.view(b, out.frame_count, tpf, self.d_model)
            frame_ids = torch.arange(out.frame_count, device=tokens.device, dtype=torch.long)
            tokens = tokens + self.frame_emb(frame_ids).view(1, out.frame_count, 1, self.d_model)
            tokens = tokens.view(b, out.frame_count * tpf, self.d_model)

        if self.feature_token_builder is not None:
            feat_tokens = self.feature_token_builder(features)
            if feat_tokens is not None:
                tokens = torch.cat([tokens, feat_tokens], dim=1)
                if mask is not None:
                    pad = torch.zeros(
                        (b, feat_tokens.shape[1]), device=mask.device, dtype=torch.bool
                    )
                    mask = torch.cat([mask, pad], dim=1)

        if self.use_cls_token:
            cls = self.cls_token.expand(b, 1, self.d_model)  # type: ignore[union-attr]
            tokens = torch.cat([cls, tokens], dim=1)
            if mask is not None:
                cls_mask = torch.zeros((b, 1), device=mask.device, dtype=torch.bool)
                mask = torch.cat([cls_mask, mask], dim=1)

        tokens = self.mixer(tokens, mask)

        if self.pooling == "flatten":
            if self.use_cls_token:
                tokens = tokens[:, 1:, :]
            flat = tokens.reshape(b, -1)
            return self.head(flat)

        pooled = pool_tokens(
            tokens,
            pooling=self.pooling,
            has_cls=self.use_cls_token,
            token_mask=mask,
            mask_pool=self.mask_pool,
        )
        return self.head(pooled)
