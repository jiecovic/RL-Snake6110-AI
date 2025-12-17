# src/snake_rl/game/analysis/pathfinding.py
from __future__ import annotations

from typing import List, Optional

import networkx as nx

from snake_rl.game.geometry import Point
from snake_rl.game.snakegame import SnakeGame


def shortest_path_to_closest_food_dijkstra(game: SnakeGame) -> Optional[List[Point]]:
    """
    Uses Dijkstra/shortest_path via networkx to find shortest path from head to closest food.
    Ignores moving tail; excludes snake body (except head).
    """
    head = game.get_head_position()

    G = nx.Graph()
    for y in range(game.height):
        for x in range(game.width):
            p = Point(x, y)

            if p in game.wall_positions or (p in game.snake and p != head):
                continue

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < game.width and 0 <= ny_ < game.height:
                    neighbor = Point(nx_, ny_)
                    if neighbor in game.wall_positions:
                        continue
                    if neighbor in game.snake and neighbor != head:
                        continue
                    G.add_edge(p, neighbor)

    paths: list[list[Point]] = []
    for food in game.food:
        try:
            path = nx.shortest_path(G, source=head, target=food)
            paths.append(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

    if not paths:
        return None

    return min(paths, key=len)
