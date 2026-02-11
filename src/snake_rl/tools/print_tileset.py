# src\snake_rl\tools\print_tileset.py
from __future__ import annotations

from snake_rl.game.tile_types import TileType
from snake_rl.game.tileset import Tileset


def main() -> None:
    try:
        tileset = Tileset()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ Failed to load tileset: {e}")
        return

    print(f"✅ Loaded tileset from: {tileset.filepath}")
    print(f"Tile size: {tileset.tile_size}")
    print()

    for tile_type in TileType:
        if tile_type in tileset:
            print(f"🧩 Tile: {tile_type.name}")
            tile = tileset[tile_type]
            for row in tile:
                print("".join(str(x) for x in row))
            print()
        else:
            print(f"⚠️  Tile missing: {tile_type.name}")


if __name__ == "__main__":
    main()
