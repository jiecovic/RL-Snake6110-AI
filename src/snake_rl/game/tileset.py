# src/snake_rl/game/tileset.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .tile_types import TileType


class Tileset:
    """
    YAML-based tileset loader.

    Expected YAML schema:

        tile_size: int
        tiles:
          <tile_name>:
            glyph: "..."          # optional, for debug / ASCII rendering / parsing
            pixels: [[int]]       # required, tile_size x tile_size

    Design notes:
    - Tiles are keyed by TileType (enum).
    - Glyphs are stored and we also build a reverse mapping glyph -> [TileType,...].
      This is needed now because glyph mapping is owned by the tileset YAML (not tile_types.py).
    - Duplicate glyphs are allowed (e.g. vertical_up + vertical_down may both use '║').
      For parsing, we provide a "canonical" choice: the first tile encountered for that glyph.
      (This is fine because levels will soon be static-only and shouldn't contain snake glyphs anyway.)
    - All TileTypes MUST be present in the tileset.
      (Intentional strictness: missing visual definitions should fail loudly.)
    """

    def __init__(self, filepath: str = "assets/tilesets/classic.yaml"):
        self.filepath = Path(filepath)
        self.tile_size: int
        self.tiles: Dict[TileType, List[List[int]]] = {}

        # TileType -> glyph (optional)
        self.glyphs: Dict[TileType, str] = {}

        # glyph -> list of TileTypes (duplicates allowed)
        self.glyph_to_tiles: Dict[str, List[TileType]] = {}

        # glyph -> canonical TileType (first encountered)
        self.glyph_to_tile: Dict[str, TileType] = {}

        data = self._read_yaml(self.filepath)
        self._load(data)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    def _read_yaml(self, path: Path) -> dict:
        if not path.is_file():
            raise FileNotFoundError(f"Tileset file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise TypeError(
                f"Tileset YAML must be a mapping, got {type(data).__name__}: {path}"
            )
        return data

    def _load(self, data: dict) -> None:
        tile_size = data.get("tile_size")
        if not isinstance(tile_size, int) or tile_size <= 0:
            raise ValueError(f"Invalid or missing 'tile_size' in {self.filepath}")
        self.tile_size = tile_size

        raw_tiles = data.get("tiles")
        if not isinstance(raw_tiles, dict):
            raise ValueError(f"Missing or invalid 'tiles' section in {self.filepath}")

        defined_types = set()

        # Clear (in case someone reuses the instance in the future)
        self.tiles.clear()
        self.glyphs.clear()
        self.glyph_to_tiles.clear()
        self.glyph_to_tile.clear()

        for name, spec in raw_tiles.items():
            if not isinstance(spec, dict):
                raise TypeError(
                    f"Tile '{name}' must be a mapping with keys like 'glyph' and 'pixels'"
                )

            try:
                tile_type = TileType[name.upper()]
            except KeyError as e:
                valid = ", ".join(t.name for t in TileType)
                raise ValueError(
                    f"Unknown tile name '{name}' in tileset. "
                    f"Valid TileTypes: {valid}"
                ) from e

            pixels = spec.get("pixels")
            if not self._is_valid_matrix(pixels, self.tile_size):
                raise ValueError(
                    f"Invalid or inconsistent 'pixels' matrix for tile '{name}' "
                    f"(expected {self.tile_size}x{self.tile_size})"
                )

            # Optional: validate pixels are ints (YAML can load weird scalars)
            if not self._pixels_are_ints(pixels):
                raise ValueError(f"Tile '{name}' has non-integer pixel values")

            glyph = spec.get("glyph")
            if glyph is not None:
                if not isinstance(glyph, str) or len(glyph) != 1:
                    raise ValueError(
                        f"Invalid glyph for tile '{name}': must be a single-character string"
                    )
                self.glyphs[tile_type] = glyph

                # Build reverse map (duplicates allowed)
                self.glyph_to_tiles.setdefault(glyph, []).append(tile_type)

                # Canonical mapping = first encountered
                self.glyph_to_tile.setdefault(glyph, tile_type)

            self.tiles[tile_type] = pixels
            defined_types.add(tile_type)

        # Ensure EMPTY exists (even if author forgot it)
        if TileType.EMPTY not in self.tiles:
            self.tiles[TileType.EMPTY] = [
                [0 for _ in range(self.tile_size)]
                for _ in range(self.tile_size)
            ]
            self.glyphs.setdefault(TileType.EMPTY, " ")
            self.glyph_to_tiles.setdefault(" ", []).append(TileType.EMPTY)
            self.glyph_to_tile.setdefault(" ", TileType.EMPTY)
            defined_types.add(TileType.EMPTY)

        # Enforce completeness: every TileType must be defined
        missing = set(TileType) - defined_types
        if missing:
            names = ", ".join(t.name for t in sorted(missing, key=lambda x: x.value))
            raise ValueError(
                f"Tileset is missing definitions for TileTypes: {names}"
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _is_valid_matrix(self, matrix: List[List[int]] | None, size: int) -> bool:
        if not isinstance(matrix, list) or len(matrix) != size:
            return False
        return all(isinstance(row, list) and len(row) == size for row in matrix)

    def _pixels_are_ints(self, matrix: List[List[int]]) -> bool:
        for row in matrix:
            for v in row:
                if not isinstance(v, int):
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #

    def __getitem__(self, tile_type: TileType) -> List[List[int]]:
        return self.tiles[tile_type]

    def __contains__(self, tile_type: TileType) -> bool:
        return tile_type in self.tiles

    def glyph_for(self, tile_type: TileType) -> str | None:
        """Optional helper for debug rendering."""
        return self.glyphs.get(tile_type)

    def tile_for_glyph(self, glyph: str) -> TileType | None:
        """
        Return a canonical TileType for a glyph (first encountered in YAML).

        NOTE:
        - If multiple TileTypes share a glyph (expected for e.g. left/right variants),
          this returns a stable canonical representative.
        - For strict parsing, use `tiles_for_glyph()` and handle ambiguity yourself.
        """
        if not isinstance(glyph, str) or len(glyph) != 1:
            raise ValueError("glyph must be a single character")
        return self.glyph_to_tile.get(glyph)

    def tiles_for_glyph(self, glyph: str) -> List[TileType]:
        """Return all TileTypes registered for a glyph (possibly empty)."""
        if not isinstance(glyph, str) or len(glyph) != 1:
            raise ValueError("glyph must be a single character")
        return list(self.glyph_to_tiles.get(glyph, []))
