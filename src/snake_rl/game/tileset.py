import json
from typing import Dict, List
from .tile_types import TileType


class Tileset:
    def __init__(self, filepath: str ="assets/tilesets/classic.json"):
        self.filepath = filepath
        self.tile_size: int
        self.tiles: Dict[TileType, List[List[int]]] = {}

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._load(data)

    def _load(self, data: dict):
        tile_size = data.get("tile_size")
        if not isinstance(tile_size, int) or tile_size <= 0:
            raise ValueError(f"Invalid or missing 'tile_size' in {self.filepath}")
        self.tile_size = tile_size

        raw_tiles = data.get("tiles")
        if not isinstance(raw_tiles, dict):
            raise ValueError(f"Missing or invalid 'tiles' section in {self.filepath}")

        defined_types = set()

        for name, matrix in raw_tiles.items():
            try:
                tile_type = TileType[name.upper()]
            except KeyError:
                raise ValueError(f"Unknown tile name in tileset: '{name}'")

            if not self._is_valid_matrix(matrix, tile_size):
                raise ValueError(f"Invalid or inconsistent matrix size for tile '{name}'")

            self.tiles[tile_type] = matrix
            defined_types.add(tile_type)

        # Add fallback for EMPTY if missing
        if TileType.EMPTY not in self.tiles:
            # print("⚠️  'EMPTY' tile not defined in tileset — using default all-zero matrix.")
            self.tiles[TileType.EMPTY] = [
                [0 for _ in range(tile_size)] for _ in range(tile_size)
            ]
            defined_types.add(TileType.EMPTY)

        # Check that all other TileTypes are defined
        missing = set(TileType) - defined_types
        if missing:
            names = ", ".join(t.name for t in sorted(missing, key=lambda x: x.value))
            raise ValueError(f"Tileset is missing definitions for: {names}")

    def _is_valid_matrix(self, matrix: List[List[int]], size: int) -> bool:
        if not isinstance(matrix, list) or len(matrix) != size:
            return False
        return all(isinstance(row, list) and len(row) == size for row in matrix)

    def __getitem__(self, tile_type: TileType) -> List[List[int]]:
        return self.tiles[tile_type]

    def __contains__(self, tile_type: TileType) -> bool:
        return tile_type in self.tiles
