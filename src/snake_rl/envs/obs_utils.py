# src/snake_rl/envs/obs_utils.py
from __future__ import annotations

from typing import Literal, overload

import numpy as np

from snake_rl.envs.view_radius import parse_view_radius
from snake_rl.game.geometry import Direction, Point
from snake_rl.game.tile_types import TileType


def _rotate_tile_block(block: np.ndarray, d: Direction) -> np.ndarray:
    """
    Rotate a tile's pixel block so the egocentric view has "forward == UP".
    """
    if d == Direction.UP:
        k = 0
    elif d == Direction.RIGHT:
        k = 1
    elif d == Direction.DOWN:
        k = 2
    elif d == Direction.LEFT:
        k = 3
    else:
        k = 0

    if k == 0:
        return block
    return np.rot90(block, k=k).copy()


def global_pixel_frame(
    *,
    pixel_grid: np.ndarray,
    tile_size: int,
    remove_border: bool,
) -> np.ndarray:
    if not remove_border:
        return pixel_grid
    ts = int(tile_size)
    return pixel_grid[ts:-ts, ts:-ts]


@overload
def pov_pixel_frame(
    *,
    pixel_grid: np.ndarray,
    tile_size: int,
    head: Point,
    direction: Direction,
    view_radius: int | tuple[int, int],
    rotate_to_head: bool = True,
    oob_fill_value: int = 0,
    return_valid: Literal[True],
) -> tuple[np.ndarray, np.ndarray]: ...


@overload
def pov_pixel_frame(
    *,
    pixel_grid: np.ndarray,
    tile_size: int,
    head: Point,
    direction: Direction,
    view_radius: int | tuple[int, int],
    rotate_to_head: bool = True,
    oob_fill_value: int = 0,
    return_valid: Literal[False] = False,
) -> np.ndarray: ...


def pov_pixel_frame(
    *,
    pixel_grid: np.ndarray,
    tile_size: int,
    head: Point,
    direction: Direction,
    view_radius: int | tuple[int, int],
    rotate_to_head: bool = True,
    oob_fill_value: int = 0,
    return_valid: bool = False,
):
    tilesize = int(tile_size)
    ry, rx = parse_view_radius(view_radius)
    view_tiles_y = 2 * ry + 1
    view_tiles_x = 2 * rx + 1
    view_h = view_tiles_y * tilesize
    view_w = view_tiles_x * tilesize

    head_x, head_y = int(head.x), int(head.y)
    grid_w = pixel_grid.shape[1] // tilesize
    grid_h = pixel_grid.shape[0] // tilesize

    fill = np.uint8(int(oob_fill_value) & 0xFF)
    vision = np.full((view_h, view_w), fill, dtype=np.uint8)

    valid: np.ndarray | None = None
    if return_valid:
        valid = np.zeros((view_h, view_w), dtype=np.bool_)

    if not rotate_to_head:
        col_start = (head_x - rx) * tilesize
        col_end = (head_x + rx + 1) * tilesize
        row_start = (head_y - ry) * tilesize
        row_end = (head_y + ry + 1) * tilesize

        grid_row_start = max(0, row_start)
        grid_row_end = min(pixel_grid.shape[0], row_end)
        grid_col_start = max(0, col_start)
        grid_col_end = min(pixel_grid.shape[1], col_end)

        vision_row_start = grid_row_start - row_start
        vision_row_end = vision_row_start + (grid_row_end - grid_row_start)
        vision_col_start = grid_col_start - col_start
        vision_col_end = vision_col_start + (grid_col_end - grid_col_start)

        if grid_row_end > grid_row_start and grid_col_end > grid_col_start:
            grid_view = pixel_grid[
                grid_row_start:grid_row_end,
                grid_col_start:grid_col_end,
            ].astype(np.uint8, copy=False)
            vision[vision_row_start:vision_row_end, vision_col_start:vision_col_end] = grid_view
            if valid is not None:
                valid[vision_row_start:vision_row_end, vision_col_start:vision_col_end] = True

        if return_valid:
            assert valid is not None
            return vision, valid
        return vision

    d = direction

    for oy in range(-ry, ry + 1):
        for ox in range(-rx, rx + 1):
            dx_ego = ox
            dy_ego = oy

            if d == Direction.UP:
                dx_w = dx_ego
                dy_w = dy_ego
            elif d == Direction.RIGHT:
                dx_w = -dy_ego
                dy_w = dx_ego
            elif d == Direction.DOWN:
                dx_w = -dx_ego
                dy_w = -dy_ego
            elif d == Direction.LEFT:
                dx_w = dy_ego
                dy_w = -dx_ego
            else:
                dx_w = dx_ego
                dy_w = dy_ego

            wx = head_x + dx_w
            wy = head_y + dy_w

            if not (0 <= wx < grid_w and 0 <= wy < grid_h):
                continue

            src_y0 = wy * tilesize
            src_y1 = src_y0 + tilesize
            src_x0 = wx * tilesize
            src_x1 = src_x0 + tilesize

            dst_y0 = (oy + ry) * tilesize
            dst_y1 = dst_y0 + tilesize
            dst_x0 = (ox + rx) * tilesize
            dst_x1 = dst_x0 + tilesize

            block = pixel_grid[src_y0:src_y1, src_x0:src_x1].astype(np.uint8, copy=False)
            block = _rotate_tile_block(block, d)

            vision[dst_y0:dst_y1, dst_x0:dst_x1] = block
            if valid is not None:
                valid[dst_y0:dst_y1, dst_x0:dst_x1] = True

    if return_valid:
        assert valid is not None
        return vision, valid
    return vision


def global_tile_frame(*, tile_grid: np.ndarray, remove_border: bool) -> np.ndarray:
    if not remove_border:
        return tile_grid
    return tile_grid[1:-1, 1:-1]


def pov_tile_frame_with_valid(
    *,
    tile_grid: np.ndarray,
    head: Point,
    direction: Direction,
    view_radius: int | tuple[int, int],
    rotate_to_head: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    g = tile_grid

    hx, hy = int(head.x), int(head.y)
    ry, rx = parse_view_radius(view_radius)
    vy = 2 * ry + 1
    vx = 2 * rx + 1

    out = np.full((vy, vx), int(TileType.EMPTY.value), dtype=np.uint8)
    valid = np.zeros((vy, vx), dtype=np.bool_)

    if not rotate_to_head:
        x0, x1 = hx - rx, hx + rx + 1
        y0, y1 = hy - ry, hy + ry + 1

        gy0 = max(0, y0)
        gy1 = min(g.shape[0], y1)
        gx0 = max(0, x0)
        gx1 = min(g.shape[1], x1)

        oy0 = gy0 - y0
        oy1 = oy0 + (gy1 - gy0)
        ox0 = gx0 - x0
        ox1 = ox0 + (gx1 - gx0)

        out[oy0:oy1, ox0:ox1] = g[gy0:gy1, gx0:gx1]
        valid[oy0:oy1, ox0:ox1] = True
        return out, valid

    d = direction
    ys = np.arange(-ry, ry + 1, dtype=np.int32)
    xs = np.arange(-rx, rx + 1, dtype=np.int32)
    dy_ego, dx_ego = np.meshgrid(ys, xs, indexing="ij")

    if d == Direction.UP:
        dx_w = dx_ego
        dy_w = dy_ego
    elif d == Direction.RIGHT:
        dx_w = -dy_ego
        dy_w = dx_ego
    elif d == Direction.DOWN:
        dx_w = -dx_ego
        dy_w = -dy_ego
    elif d == Direction.LEFT:
        dx_w = dy_ego
        dy_w = -dx_ego
    else:
        dx_w = dx_ego
        dy_w = dy_ego

    xw = hx + dx_w
    yw = hy + dy_w

    mask = (xw >= 0) & (xw < g.shape[1]) & (yw >= 0) & (yw < g.shape[0])
    out[mask] = g[yw[mask], xw[mask]]
    valid[mask] = True
    return out, valid


__all__ = [
    "global_pixel_frame",
    "pov_pixel_frame",
    "global_tile_frame",
    "pov_tile_frame_with_valid",
]
