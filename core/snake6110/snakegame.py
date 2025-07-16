import pygame
import random
import numpy as np
from typing import Optional, List, Union, Dict, Set
from enum import Enum, auto
from collections import deque
import networkx as nx
from core.snake6110.level import BaseLevel
from core.snake6110.geometry import Point, Direction, DIRECTION_TO_POINT, RelativeDirection
from core.snake6110.tile_types import is_wall, is_food, is_tail, TileType
from core.snake6110.tileset import Tileset


class MoveResult(Enum):
    OK = auto()
    FOOD_EATEN = auto()
    HIT_BOUNDARY = auto()
    HIT_WALL = auto()
    HIT_SELF = auto()
    GAME_NOT_RUNNING = auto()
    TIMEOUT = auto()
    CYCLE_DETECTED = auto()
    WIN = auto()  # ✅ Snake filled the entire grid


class SnakeGame:
    def __init__(
            self,
            level: BaseLevel,
            food_count: Optional[int] = None,
            fps: int = 1,
            pixel_size: int = 10,
            tileset: Optional[Tileset] = None,
            seed: Optional[int] = None,
    ):
        # === Basic config ===
        self.level = level
        self.width = level.width
        self.height = level.height
        self.pixel_size = pixel_size
        self.fps = fps
        self.tileset = tileset or Tileset()

        # === RNG ===
        self.seed_value = seed
        self.rng = random.Random(seed)

        # === Rendering ===
        self.tile_dim = self.tileset.tile_size * self.pixel_size
        self.pixel_buffer = np.zeros(
            (self.height * self.tileset.tile_size, self.width * self.tileset.tile_size),
            dtype=np.uint8
        )
        self.render_enabled = False
        self.pygame_initialized = False
        self.screen = None
        self.clock = None
        self.font = None

        # === Wall setup (static) ===
        self.wall_tiles = level.get_wall_tiles()
        self.wall_positions = {pos for pos, _ in self.wall_tiles}
        self.wall_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        for pos in self.wall_positions:
            self.wall_layer[pos.y, pos.x] = 1

        # === Layers ===
        self.snake_layers: Dict[TileType, np.ndarray] = {}
        self.food_layer = np.zeros((self.height, self.width), dtype=np.uint8)

        # === Food target count ===
        existing_food = level.get_food_positions()
        if food_count is not None:
            self.target_food_count = food_count
        elif existing_food:
            self.target_food_count = len(existing_food)
        else:
            raise ValueError("No food defined in level and no food_count specified.")

        # === Game state trackers (used in reset) ===
        self.spawnable_tiles: Set[Point] = set()
        self.snake: List[Point] = []
        self.food: List[Point] = []
        self.direction = None

        self.visited = set()
        self.recent_heads = deque(maxlen=self.max_playable_tiles)

        # === Run reset ===
        self.reset()

    def reset(self) -> None:
        """Reset snake, food, score, state flags, and render layers."""
        # === Game state ===
        self.snake = self.level.get_snake_segments()
        self.direction = self.level.get_snake_direction()
        self.food = list(self.level.get_food_positions())
        self.score = 0
        self.running = True

        # === Available points ===
        self.spawnable_tiles = {
            Point(x, y)
            for y in range(self.height)
            for x in range(self.width)
            if Point(x, y) not in self.wall_positions
        }
        self.spawnable_tiles -= set(self.snake)
        self.spawnable_tiles -= set(self.food)

        # === Food fill ===
        self._spawn_food()

        # === Clear trackers ===
        self.visited.clear()
        self.recent_heads.clear()

        # === Clear and update layers ===
        self.pixel_buffer.fill(0)
        self._update_wall_layer()
        self._update_snake_layers()
        self._update_food_layer()
        self._update_pixel_buffer()

    def set_seed(self, seed: Optional[int]) -> None:
        """Set or reset the RNG for reproducibility."""
        self.seed_value = seed
        self.rng = random.Random(seed)

    @property
    def max_playable_tiles(self) -> int:
        """
        Returns the number of tiles that are not walls.
        """
        return self.width * self.height - len(self.wall_positions)

    @property
    def won(self) -> bool:
        return len(self.snake) >= self.max_playable_tiles

    def get_head_position(self) -> Point:
        """
        Returns the current position of the snake's head.
        """
        return self.snake[0]

    def get_food_positions(self) -> List[Point]:
        """
        Returns the current list of food positions.
        """
        return self.food.copy()

    def _spawn_food(self) -> int:
        """
        Attempts to spawn food until reaching the target_food_count.
        Returns the number of food tiles successfully placed.
        """
        need = self.target_food_count - len(self.food)
        if need <= 0 or not self.spawnable_tiles:
            return 0

        candidates = self.rng.sample(list(self.spawnable_tiles), k=min(need, len(self.spawnable_tiles)))

        for p in candidates:
            self.food.append(p)
            self.spawnable_tiles.remove(p)

        return len(candidates)



    def move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> List[MoveResult]:
        result = self._move(rel_dir)
        self._update_snake_layers()
        self._update_pixel_buffer()
        return result

    def _move(self, rel_dir: RelativeDirection = RelativeDirection.FORWARD) -> List[MoveResult]:
        if not self.running:
            return [MoveResult.GAME_NOT_RUNNING]

        results = []

        # === 1. Determine new direction and target ===
        new_direction = self.direction
        if rel_dir == RelativeDirection.LEFT:
            new_direction = self.direction.turn_left()
        elif rel_dir == RelativeDirection.RIGHT:
            new_direction = self.direction.turn_right()

        vec = DIRECTION_TO_POINT[new_direction]
        new_head = Point(self.snake[0].x + vec.x, self.snake[0].y + vec.y)

        # === 2. Collision checks ===
        if not (0 <= new_head.x < self.width and 0 <= new_head.y < self.height):
            self.running = False
            return [MoveResult.HIT_BOUNDARY]

        if new_head in self.wall_positions:
            self.running = False
            return [MoveResult.HIT_WALL]

        if new_head in self.snake:
            self.running = False
            return [MoveResult.HIT_SELF]

        # === 3. Apply movement ===
        self.direction = new_direction
        self.snake.insert(0, new_head)
        self.spawnable_tiles.discard(new_head)
        self.recent_heads.append(new_head)
        self.visited.add(new_head)

        # === 4. Food consumption or tail removal ===
        if new_head in self.food:
            self.food.remove(new_head)
            self.score += 1
            self.recent_heads.clear()
            self.visited.clear()
            results.append(MoveResult.FOOD_EATEN)

            self._spawn_food()

            # === Win check ===
            if len(self.food) == 0 and self.won:
                self.running = False
                results.append(MoveResult.WIN)
        else:
            tail = self.snake.pop()
            self.spawnable_tiles.add(tail)

        # === 5. Cycle detection (non-fatal) ===
        if self.has_head_cycle():
            results.append(MoveResult.CYCLE_DETECTED)

        results.insert(0, MoveResult.OK)
        return results

    def _update_wall_layer(self) -> None:
        """Update the binary wall layer based on static wall positions."""
        self.wall_layer = np.zeros((self.height, self.width), dtype=np.uint8)
        for pos in self.wall_positions:
            self.wall_layer[pos.y, pos.x] = 1

    def _update_food_layer(self) -> None:
        self.food_layer.fill(0)
        for food in self.food:
            self.food_layer[food.y, food.x] = 1

    def _update_snake_layers(self) -> None:
        self.snake_layers = {
            tile_type: np.zeros((self.height, self.width), dtype=np.float32)
            for tile_type in TileType
            if tile_type.name.startswith("SNAKE_")
        }

        if len(self.snake) < 2:
            return

        # Head
        head = self.snake[0]
        head_type = {
            Direction.UP: TileType.SNAKE_HEAD_UP,
            Direction.DOWN: TileType.SNAKE_HEAD_DOWN,
            Direction.LEFT: TileType.SNAKE_HEAD_LEFT,
            Direction.RIGHT: TileType.SNAKE_HEAD_RIGHT
        }[self.direction]
        self.snake_layers[head_type][head.y, head.x] = 1.0

        # Body
        for i in range(1, len(self.snake) - 1):
            prev = self.snake[i - 1]
            curr = self.snake[i]
            next_ = self.snake[i + 1]

            if prev.x == next_.x:
                tile_type = TileType.SNAKE_BODY_VERTICAL
            elif prev.y == next_.y:
                tile_type = TileType.SNAKE_BODY_HORIZONTAL
            elif ((prev.x < curr.x and next_.y > curr.y) or
                  (next_.x < curr.x and prev.y > curr.y)):
                tile_type = TileType.SNAKE_BODY_TR
            elif ((prev.x > curr.x and next_.y > curr.y) or
                  (next_.x > curr.x and prev.y > curr.y)):
                tile_type = TileType.SNAKE_BODY_TL
            elif ((prev.x < curr.x and next_.y < curr.y) or
                  (next_.x < curr.x and prev.y < curr.y)):
                tile_type = TileType.SNAKE_BODY_BR
            elif ((prev.x > curr.x and next_.y < curr.y) or
                  (next_.x > curr.x and prev.y < curr.y)):
                tile_type = TileType.SNAKE_BODY_BL
            else:
                raise RuntimeError(f"Could not determine body tile type at index {i}: prev={prev}, curr={curr}, next={next_}")

            self.snake_layers[tile_type][curr.y, curr.x] = 1.0

        # Tail
        tail = self.snake[-1]
        prev = self.snake[-2]
        if prev.x < tail.x:
            tail_type = TileType.SNAKE_TAIL_RIGHT
        elif prev.x > tail.x:
            tail_type = TileType.SNAKE_TAIL_LEFT
        elif prev.y < tail.y:
            tail_type = TileType.SNAKE_TAIL_DOWN
        elif prev.y > tail.y:
            tail_type = TileType.SNAKE_TAIL_UP
        else:
            raise RuntimeError(f"Could not determine tail direction: prev={prev}, tail={tail}")

        self.snake_layers[tail_type][tail.y, tail.x] = 1.0

    def get_all_layers(self, channel_last: bool = False, merge_snake_layers: bool = False) -> np.ndarray:
        """
        Returns all game layers (snake, wall, food) as a stacked NumPy array.

        Args:
            channel_last (bool): If True, returns shape (H, W, C). Else, (C, H, W).
            merge_snake_layers (bool): If True, merges all snake layers into a single channel.

        Returns:
            np.ndarray: The combined layer tensor.
        """
        if merge_snake_layers:
            snake_combined = np.zeros((self.height, self.width), dtype=np.float32)
            for layer in self.snake_layers.values():
                snake_combined += layer
            layers = [snake_combined]
        else:
            layers = list(self.snake_layers.values())

        layers.append(self.wall_layer)
        layers.append(self.food_layer)

        stacked = np.stack(layers, axis=0)  # shape: (C, H, W)

        if channel_last:
            stacked = np.transpose(stacked, (1, 2, 0))  # shape: (H, W, C)

        return stacked

    def _update_pixel_buffer(self) -> None:
        """
        Updates the pixel buffer (`self.pixel_buffer`) as a 2D NumPy array of shape
        (height * tile_size, width * tile_size), where each tile is drawn without additional upscaling.
        """
        tile_dim = self.tileset.tile_size
        pixel_height = self.height * tile_dim
        pixel_width = self.width * tile_dim

        self.pixel_buffer = np.zeros((pixel_height, pixel_width), dtype=np.uint8)

        # === Draw walls ===
        for point, tile_type in self.wall_tiles:
            if tile_type in self.tileset:
                tile = np.array(self.tileset[tile_type], dtype=np.uint8)
                py, px = point.y * tile_dim, point.x * tile_dim
                self.pixel_buffer[py:py + tile_dim, px:px + tile_dim] = tile

        # === Draw trail ===
        # if TileType.TRACKER_VISITED in self.tileset:
        #     tile = np.array(self.tileset[TileType.TRACKER_VISITED], dtype=np.uint8)
        #     for point in self.visited:
        #         py, px = point.y * tile_dim, point.x * tile_dim
        #         self.pixel_buffer[py:py + tile_dim, px:px + tile_dim] = tile
        #

        # === Draw path to food ===
        # if self.path_to_food and TileType.TRACKER_PATH in self.tileset:
        #     path_tile = np.array(self.tileset[TileType.TRACKER_PATH], dtype=np.uint8)
        #     for point in self.path_to_food[1:]:  # Skip head (already drawn)
        #         py, px = point.y * tile_dim, point.x * tile_dim
        #         self.pixel_buffer[py:py + tile_dim, px:px + tile_dim] = path_tile

        # === Draw food ===
        if TileType.FOOD in self.tileset:
            tile = np.array(self.tileset[TileType.FOOD], dtype=np.uint8)
            for point in self.food:
                py, px = point.y * tile_dim, point.x * tile_dim
                self.pixel_buffer[py:py + tile_dim, px:px + tile_dim] = tile

        # === Draw snake ===
        for tile_type, layer in self.snake_layers.items():
            if tile_type in self.tileset:
                tile = np.array(self.tileset[tile_type], dtype=np.uint8)
                yxs = np.argwhere(layer > 0)
                for y, x in yxs:
                    py, px = y * tile_dim, x * tile_dim
                    self.pixel_buffer[py:py + tile_dim, px:px + tile_dim] = tile

    def _init_renderer(self):
        if self.pygame_initialized:
            return
        pygame.init()
        width = self.width * self.tile_dim
        height = self.height * self.tile_dim
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.pygame_initialized = True
        self.render_enabled = True

    def render(self) -> None:
        """
        Public method to render the game. Initializes pygame lazily.
        """
        if not self.pygame_initialized:
            self._init_renderer()

        if self.pixel_buffer is None:
            return  # nothing to draw yet

        self._draw()

        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        self.clock.tick(self.fps)

    def _draw(self) -> None:
        if self.pixel_buffer is None:
            return

        # 1. Upscale pixel_buffer (2D grayscale) using np.kron
        upscaled = np.kron(
            self.pixel_buffer,
            np.ones((self.pixel_size, self.pixel_size), dtype=self.pixel_buffer.dtype)
        )

        # 2. Convert to RGB by stacking the grayscale values 3 times
        rgb_buffer = np.stack([upscaled * 255] * 3, axis=-1).astype(np.uint8)

        # 3. Convert to surface and draw
        surface = pygame.surfarray.make_surface(rgb_buffer.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))

        # 4. Draw score and FPS
        score_text = f"Score: {self.score}"
        fps_text = f"FPS: {self.clock.get_fps():.2f}"
        combined_text = f"{score_text}   {fps_text}"
        info_surf = self.font.render(combined_text, True, (255, 255, 255))
        self.screen.blit(info_surf, (10, 10))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    # === Experimental: Pathfinding, Cycle Detection, etc. ===
    def find_path_to_closest_food_dijkstra(self) -> Optional[List[Point]]:
        """
        Uses Dijkstra's algorithm via networkx to find the shortest path from the head to the closest food.
        Ignores moving tail, does not anticipate future snake growth.

        Returns:
            A list of Points representing the shortest path from head to food (including head), or None if no path exists.
        """
        head = self.get_head_position()

        # 1. Build graph of walkable tiles
        G = nx.Graph()
        for y in range(self.height):
            for x in range(self.width):
                p = Point(x, y)

                # Allow the head to be part of the graph, but exclude other snake segments
                if p in self.wall_positions or (p in self.snake and p != head):
                    continue

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < self.width and 0 <= ny_ < self.height:
                        neighbor = Point(nx_, ny_)
                        if neighbor in self.wall_positions:
                            continue
                        if neighbor in self.snake and neighbor != head:
                            continue
                        G.add_edge(p, neighbor)

        # 2. Find shortest path to the closest food (if any reachable)
        paths = []
        for food in self.food:
            try:
                path = nx.shortest_path(G, source=head, target=food)
                paths.append(path)
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue

        if not paths:
            return None

        # 3. Return the shortest path among all
        return min(paths, key=len)

    def has_head_cycle(self) -> bool:
        """
        Detects if the snake has entered a repeating cycle.
        Uses Floyd's Tortoise and Hare algorithm to detect repetition,
        then verifies the actual movement pattern.
        """
        path = list(self.recent_heads)
        n = len(path)

        if n < 8:  # need at least 2 * min_cycle_len
            return False

        # === 1. Fast scan with Floyd's cycle detection ===
        tortoise = 0
        hare = 1
        while hare < n:
            if path[tortoise] == path[hare]:
                cycle_len = hare - tortoise
                if cycle_len < 4 or hare + cycle_len > n:
                    return False  # too short or not enough data to verify

                # === 2. Confirm cycle by comparing the repeated segments ===
                segment1 = path[hare - cycle_len:hare]
                segment2 = path[hare:hare + cycle_len]
                if segment1 == segment2:
                    return True  # confirmed cycle

            tortoise += 1
            hare += 2

        return False
