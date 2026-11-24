"""Level metrics for the expressive-range report.

The generator already records ``path_ledges``, ``rooms_on_path``, ``min_window``
and ``mean_window`` in ``metadata.stats`` (they are cheap and known exactly at
generation time). This module derives the geometry metrics (linearity, leniency,
density, backtrack ratio) from the emitted tiles and metadata.
"""
from __future__ import annotations

from dataclasses import dataclass

from game import Level


@dataclass(frozen=True)
class LevelMetrics:
    path_ledges: int
    rooms_on_path: int
    min_window: float
    mean_window: float
    linearity: float
    leniency: float
    density: float
    backtrack_ratio: float


def _find(tiles: tuple[str, ...], ch: str):
    for r, row in enumerate(tiles):
        c = row.find(ch)
        if c >= 0:
            return c, r
    return None


def _linearity(points: list[tuple[float, float]], rows: int) -> float:
    """1 - (mean |row deviation| from the least-squares line) / rows."""
    n = len(points)
    if n < 2:
        return 1.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        slope, intercept = 0.0, my
    else:
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        intercept = my - slope * mx
    dev = sum(abs(y - (slope * x + intercept)) for x, y in zip(xs, ys)) / n
    return max(0.0, 1.0 - dev / rows)


def metrics(level: Level) -> LevelMetrics:
    md = level.metadata
    stats = md.get('stats', {})
    waypoints = md.get('waypoints', [])
    tiles = level.tiles
    rows = len(tiles)
    cols = len(tiles[0]) if rows else 0

    # density: fraction of solid tiles
    solid = sum(row.count('#') for row in tiles)
    density = solid / (rows * cols) if rows and cols else 0.0

    # linearity: how straight the critical path is (waypoint centres)
    pts = [((w[1] + w[2]) / 2.0, float(w[0])) for w in waypoints]
    linearity = _linearity(pts, rows)

    # leniency: (spike tiles + floor-pit tiles) / traversable path tiles
    spikes = sum(row.count('^') for row in tiles)
    pits = 0
    for r in (rows - 2, rows - 1):
        if 0 <= r < rows:
            pits += tiles[r].count('.')
    ledge_tiles = sum(w[2] - w[1] + 1 for w in waypoints)
    traversable = ledge_tiles + (cols * 2 - pits) + cols  # ledges + floor + headroom
    leniency = (spikes + pits) / traversable if traversable else 0.0

    # backtrack ratio: Manhattan(spawn->gem) / Manhattan(gem->door); no door => 0
    spawn, gem, door = _find(tiles, 'P'), _find(tiles, 'G'), _find(tiles, 'D')
    if spawn and gem:
        d_out = abs(spawn[0] - gem[0]) + abs(spawn[1] - gem[1])
        if door:
            d_back = abs(gem[0] - door[0]) + abs(gem[1] - door[1])
            backtrack_ratio = d_out / d_back if d_back else 0.0
        else:
            backtrack_ratio = 0.0
    else:
        backtrack_ratio = 0.0

    return LevelMetrics(
        path_ledges=stats.get('path_ledges', 0),
        rooms_on_path=stats.get('rooms_on_path', 0),
        min_window=stats.get('min_window', 0),
        mean_window=stats.get('mean_window', 0.0),
        linearity=linearity,
        leniency=leniency,
        density=density,
        backtrack_ratio=backtrack_ratio,
    )
