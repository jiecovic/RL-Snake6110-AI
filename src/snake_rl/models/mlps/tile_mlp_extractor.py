# src/snake_rl/models/mlps/tile_mlp_extractor.py
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from snake_rl.models.vits.vit_utils import GridPositionalEncoding, POS_MODES


def _infer_chw(space: spaces.Box) -> Tuple[int, int, int]:
    """
    Return (C, H, W) for supported tile-id observations.
    Supported shapes:
      - (H, W)
      - (C, H, W)   (e.g. VecFrameStack -> channels-first stacking)
    """
    if not isinstance(space, spaces.Box):
        raise TypeError(f"TileMLPExtractor expects spaces.Box, got {type(space)!r}")
    shape = space.shape
    if shape is None:
        raise ValueError("observation_space.shape is None")

    if len(shape) == 2:
        h, w = int(shape[0]), int(shape[1])
        return 1, h, w
    if len(shape) == 3:
        c, h, w = int(shape[0]), int(shape[1]), int(shape[2])
        return c, h, w
    raise ValueError(f"Unsupported observation shape {shape!r}. Expected (H,W) or (C,H,W).")


def _build_mlp(in_dim: int, hidden: Iterable[int], out_dim: int, dropout: float) -> nn.Module:
    layers: list[nn.Module] = []
    d = int(in_dim)
    for h in hidden:
        h = int(h)
        layers.append(nn.Linear(d, h))
        layers.append(nn.GELU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        d = h
    layers.append(nn.Linear(d, int(out_dim)))
    return nn.Sequential(*layers)


@torch.no_grad()
def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Mean over tokens excluding masked positions.
    x:    [B,T,D]
    mask: [B,T] bool (True => exclude)
    """
    keep = ~mask
    denom = keep.sum(dim=1).clamp(min=1).to(dtype=x.dtype)  # [B]
    keep_f = keep.to(dtype=x.dtype).unsqueeze(-1)  # [B,T,1]
    return (x * keep_f).sum(dim=1) / denom.unsqueeze(-1)


class TileMLPExtractor(BaseFeaturesExtractor):
    """
    Tile-ID embedding + (optional) positional encoding + MLP head.

    Compared to TileViTExtractor, this is "no attention": we embed per-cell tokens and then either:
      - flatten -> big MLP (classic baseline)
      - pool tokens -> smaller MLP (often more stable / less params)
      - optional CLS token -> cls / cls_mean pooling

    Input:
      obs: [B,H,W] or [B,C,H,W] with tile ids in [0, num_tiles-1]

    Pooling modes:
      - "flatten_mlp": flatten tokens then MLP (backward compatible default)
      - "mean":        mean over tokens then MLP
      - "cls":         CLS token only (requires use_cls_token=True)
      - "cls_mean":    concat [CLS, mean(tokens)] then MLP (requires use_cls_token=True)

    Token masking (optional):
      - if use_token_mask=True, tokens with tile_id == mask_token_id can be ignored for mean pooling
      - CLS token is never masked
      - masking affects "mean"/"cls_mean" only (not flatten)
    """

    POS_MODES = POS_MODES
    POOLING = {"flatten_mlp", "mean", "cls", "cls_mean"}

    def __init__(
            self,
            observation_space: spaces.Box,
            *,
            num_tiles: int,
            features_dim: int = 512,
            d_emb: int = 128,
            mlp_hidden: list[int] | tuple[int, ...] = (1024, 512),
            dropout: float = 0.0,
            # positional encoding
            pos_mode: str = "abs_2d",
            # frames
            use_frame_embed: bool = False,
            frame_fuse: str = "sum",  # "sum" or "concat"
            # normalization
            pre_ln: bool = True,
            # pooling
            pooling: str = "flatten_mlp",
            use_cls_token: bool = False,
            # token masking
            use_token_mask: bool = False,
            mask_token_id: int = 0,
            mask_pool: bool = True,
    ) -> None:
        super().__init__(observation_space, int(features_dim))

        if int(num_tiles) <= 1:
            raise ValueError(f"num_tiles must be >= 2, got {num_tiles}")
        if int(d_emb) <= 0:
            raise ValueError(f"d_emb must be > 0, got {d_emb}")
        if frame_fuse not in {"sum", "concat"}:
            raise ValueError(f"frame_fuse must be 'sum' or 'concat', got {frame_fuse!r}")

        pooling = str(pooling)
        if pooling not in self.POOLING:
            raise ValueError(f"pooling must be one of {sorted(self.POOLING)}, got {pooling!r}")
        if pooling in {"cls", "cls_mean"} and not use_cls_token:
            raise ValueError(f"pooling={pooling!r} requires use_cls_token=True")

        pos_mode = str(pos_mode)
        if pos_mode not in self.POS_MODES:
            raise ValueError(f"pos_mode must be one of {sorted(self.POS_MODES)}, got {pos_mode!r}")

        c, h, w = _infer_chw(observation_space)
        self.in_channels = int(c)
        self.h = int(h)
        self.w = int(w)
        self.seq_len = int(self.h * self.w)

        self.num_tiles = int(num_tiles)
        self.d_emb = int(d_emb)

        self.pos_mode = pos_mode
        self.use_frame_embed = bool(use_frame_embed)
        self.frame_fuse = str(frame_fuse)
        self.pre_ln = bool(pre_ln)

        self.pooling = pooling
        self.use_cls_token = bool(use_cls_token)

        self.use_token_mask = bool(use_token_mask)
        self.mask_token_id = int(mask_token_id)
        self.mask_pool = bool(mask_pool)

        # embeddings
        self.tile_emb = nn.Embedding(self.num_tiles, self.d_emb)
        nn.init.normal_(self.tile_emb.weight, mean=0.0, std=0.02)

        self.frame_emb = None
        if self.in_channels > 1 and self.use_frame_embed:
            self.frame_emb = nn.Embedding(self.in_channels, self.d_emb)
            nn.init.normal_(self.frame_emb.weight, mean=0.0, std=0.02)

        # shared grid positional encoding (same as ViT)
        self.pos_enc = GridPositionalEncoding(h=self.h, w=self.w, d_model=self.d_emb, pos_mode=self.pos_mode)

        # optional CLS token (for cls / cls_mean)
        self.cls_token = None
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_emb))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.ln = nn.LayerNorm(self.d_emb) if self.pre_ln else nn.Identity()

        # MLP head input dim depends on pooling mode
        if self.pooling == "flatten_mlp":
            if self.in_channels == 1 or self.frame_fuse == "sum":
                head_in = self.seq_len * self.d_emb
            else:
                head_in = (self.in_channels * self.seq_len) * self.d_emb
        elif self.pooling == "mean":
            head_in = self.d_emb
        elif self.pooling == "cls":
            head_in = self.d_emb
        else:  # "cls_mean"
            head_in = 2 * self.d_emb

        self.mlp = _build_mlp(head_in, mlp_hidden, int(features_dim), float(dropout))

    def _build_token_mask(self, tile_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Build a token mask aligned with the token sequence (pre-CLS).
        Returns mask: [B,T] or [B,C*T] bool, True => ignore/exclude.
        """
        if not self.use_token_mask:
            return None

        b, c, h, w = tile_ids.shape
        assert h == self.h and w == self.w

        if c == 1:
            return (tile_ids[:, 0, :, :] == self.mask_token_id).reshape(b, self.seq_len)

        if self.frame_fuse == "sum":
            m = (tile_ids == self.mask_token_id).all(dim=1)  # [B,H,W]
            return m.reshape(b, self.seq_len)

        # concat
        return (tile_ids == self.mask_token_id).reshape(b, c * self.seq_len)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 3:
            obs = observations.unsqueeze(1)  # [B,1,H,W]
        elif observations.ndim == 4:
            obs = observations
        else:
            raise ValueError(f"expected obs [B,H,W] or [B,C,H,W], got {tuple(observations.shape)}")

        b, c, h, w = obs.shape
        if h != self.h or w != self.w:
            raise ValueError(f"obs H,W mismatch: got {(h, w)} expected {(self.h, self.w)}")
        if c != self.in_channels:
            raise ValueError(f"obs C mismatch: got C={c} expected C={self.in_channels}")

        tile_ids = obs.to(dtype=torch.long)  # [B,C,H,W]
        token_mask = self._build_token_mask(tile_ids)  # [B,T] or [B,C*T] or None

        ids = tile_ids.reshape(b, c, self.seq_len)  # [B,C,T]
        t = self.tile_emb(ids)  # [B,C,T,D]

        if self.frame_emb is not None:
            frame_ids = torch.arange(c, device=obs.device, dtype=torch.long).view(1, c, 1)  # [1,C,1]
            t = t + self.frame_emb(frame_ids).expand(b, c, self.seq_len, self.d_emb)

        # fuse frames into token sequence x: [B,T,D] or [B,C*T,D]
        if c == 1:
            x = t[:, 0, :, :]
        elif self.frame_fuse == "sum":
            x = t.sum(dim=1)  # [B,T,D]
        else:
            x = t.reshape(b, c * self.seq_len, self.d_emb)  # [B,C*T,D]

        # positional encoding (apply per-frame if concat to keep same grid PE per frame)
        if c > 1 and self.frame_fuse == "concat":
            x2 = x.reshape(b, c, self.seq_len, self.d_emb)
            pos_added = []
            for fi in range(c):
                pos_added.append(self.pos_enc(x2[:, fi, :, :]))
            x = torch.cat(pos_added, dim=1)  # [B,C*T,D]
        else:
            x = self.pos_enc(x)

        x = self.ln(x)

        # If flatten mode: keep old behavior (no CLS, no mean mask handling beyond whatever you encode)
        if self.pooling == "flatten_mlp":
            flat = x.reshape(b, -1)
            return self.mlp(flat)

        # CLS handling for cls / cls_mean
        if self.cls_token is not None:
            cls = self.cls_token.expand(b, 1, self.d_emb)
            x = torch.cat([cls, x], dim=1)
            if token_mask is not None:
                cls_mask = torch.zeros((b, 1), device=token_mask.device, dtype=torch.bool)
                token_mask = torch.cat([cls_mask, token_mask], dim=1)  # [B,1+T]

        # mean / cls / cls_mean
        if self.pooling == "cls":
            pooled = x[:, 0, :]

        elif self.pooling == "mean":
            if self.cls_token is not None:
                tok = x[:, 1:, :]
                m = token_mask[:, 1:] if (token_mask is not None and self.mask_pool) else None
            else:
                tok = x
                m = token_mask if (token_mask is not None and self.mask_pool) else None

            pooled = _masked_mean(tok, m) if m is not None else tok.mean(dim=1)

        else:  # "cls_mean"
            clsv = x[:, 0, :]
            tok = x[:, 1:, :]
            m = token_mask[:, 1:] if (token_mask is not None and self.mask_pool) else None
            meanv = _masked_mean(tok, m) if m is not None else tok.mean(dim=1)
            pooled = torch.cat([clsv, meanv], dim=-1)

        return self.mlp(pooled)
