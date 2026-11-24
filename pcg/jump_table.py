"""Jump reachability measured by simulation with the real engine.

The generator never hard-codes "gap ≤ 5 tiles, drop ≤ 4". It measures what the
player can actually do by simulating take-off from a source ledge onto a target
ledge with ``game.Game``. The *window* (number of take-off frames that land on
the target) is a direct measure of execution difficulty (Compton & Mateas,
2006). If ``PhysicsConfig`` changes, the table is recomputed automatically.
"""
from __future__ import annotations

from dataclasses import dataclass

from game import Action, Game, Level
from game.constants import PLAYER_H, PLAYER_W, TILE_SIZE
from game.physics import PhysicsConfig

G_MAX = 6
D_UP = 5
D_DOWN = 8

# Micro-level geometry used for the measurement.
_ROWS, _COLS, _LEDGE_ROW = 48, 44, 30
_TAKEOFF_FRAMES = range(-1, 46)   # -1 = walk off without jumping
_SIM_FRAMES = 140


@dataclass(frozen=True)
class JumpTable:
    window: dict[tuple[int, int], int]
    walkoff: dict[tuple[int, int], bool]


def _micro_tiles(g: int, d: int) -> list[str]:
    """Source ledge (20 tiles, row 30) and a 3-tile target ledge at row 30-d."""
    grid = [['.'] * _COLS for _ in range(_ROWS)]
    for c in range(20):
        grid[_LEDGE_ROW][c] = '#'
    t0 = 20 + g
    for c in range(t0, t0 + 3):
        grid[_LEDGE_ROW - d][c] = '#'
    return [''.join(row) for row in grid]


def _lands(level: Level, physics: PhysicsConfig, f: int, t0: int, d: int) -> bool:
    """Whether take-off at frame ``f`` (or walking off when ``f == -1``) lands."""
    game = Game(level, max_steps=None, physics=physics)
    game.player.grounded = True  # the reference starts grounded on the source ledge
    for t in range(_SIM_FRAMES):
        x = game.player.x
        mv = Action.RIGHT if x < t0 * TILE_SIZE + 2 else Action.NOOP
        action = (Action.RIGHT_JUMP if mv is Action.RIGHT else Action.JUMP) if t == f else mv
        game.step(action)
        if game.is_dead:
            return False
        if (game.player.grounded and game.player.y + PLAYER_H == (_LEDGE_ROW - d) * TILE_SIZE
                and game.player.x + PLAYER_W > t0 * TILE_SIZE):
            return True
        if t > f >= 0 and game.player.grounded:
            return False  # landed back on the source ledge
    return False


def _build(physics: PhysicsConfig) -> JumpTable:
    window: dict[tuple[int, int], int] = {}
    walkoff: dict[tuple[int, int], bool] = {}
    for g in range(G_MAX + 1):
        for d in range(-D_DOWN, D_UP + 1):
            if g == 0 and d == 0:
                continue
            level = Level.from_dict({
                'version': 3,
                'tiles': _micro_tiles(g, d),
                'player_spawn': [20 * TILE_SIZE - PLAYER_W - int(1.5 * 40),
                                 _LEDGE_ROW * TILE_SIZE - PLAYER_H],
                'gem': [0, 0],
                'win_condition': 'gem',
            })
            t0 = 20 + g
            landed = 0
            for f in _TAKEOFF_FRAMES:
                hit = _lands(level, physics, f, t0, d)
                landed += hit
                if f == -1:
                    walkoff[(g, d)] = hit
            window[(g, d)] = landed
    return JumpTable(window, walkoff)


_cache: dict[PhysicsConfig, JumpTable] = {}


def build_jump_table(physics: PhysicsConfig | None = None) -> JumpTable:
    """Lazily-built, cached per ``PhysicsConfig``; ~2 s on first generation."""
    key = PhysicsConfig() if physics is None else physics
    if key not in _cache:
        _cache[key] = _build(key)
    return _cache[key]
