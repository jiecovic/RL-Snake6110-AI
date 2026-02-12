# src/snake_rl/tools/print_tileset.py
from __future__ import annotations

from snake_rl.game.snakegame import tileset_tile_names, tileset_tile_size, tileset_tiles


def main() -> None:
    tiles = tileset_tiles()
    tile_size = tileset_tile_size()

    print(f"Tile size: {tile_size}")
    print()

    names = tileset_tile_names()
    for tile_id, name in enumerate(names):
        print(f"Tile: {name}")
        tile = tiles[int(tile_id)]
        for row in tile:
            print("".join("1" if int(v) else "0" for v in row))
        print()


if __name__ == "__main__":
    main()
