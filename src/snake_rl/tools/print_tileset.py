from src.snake_rl.game.tileset import Tileset
from src.snake_rl.game.tile_types import TileType


def main():
    try:
        tileset = Tileset()
    except Exception as e:
        print(f"❌ Failed to load tileset: {e}")
        return

    print(f"✅ Loaded tileset from: {tileset.filepath}")
    print(f"Tile size: {tileset.tile_size}")
    print()

    for tile_type in TileType:
        if tile_type in tileset:
            print(f"🧩 Tile: {tile_type.name}")
            for row in tileset[tile_type]:
                print("".join(str(x) for x in row))
            print()  # blank line between tiles
        else:
            print(f"⚠️  Tile missing: {tile_type.name}")


if __name__ == "__main__":
    main()
