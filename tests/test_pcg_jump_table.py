"""pcg.jump_table: measured reachability matches the golden table exactly,
is left/right symmetric, and shrinks (never grows) for a weaker jump."""
import pytest

from game import Action, Game, Level
from game.constants import PLAYER_H, PLAYER_W, TILE_SIZE
from game.physics import PhysicsConfig
from pcg.jump_table import D_DOWN, D_UP, G_MAX, build_jump_table

# Golden table (default physics), as rows over d = -8 .. +5 (None = (0,0) skip).
_GOLDEN_ROWS = {
    0: [35, 35, 35, 35, 35, 35, 35, 35, None, 39, 36, 28, 23, 0],
    1: [35, 35, 35, 35, 35, 35, 35, 35, 37, 34, 31, 28, 23, 0],
    2: [35, 35, 35, 35, 35, 35, 35, 34, 31, 29, 26, 24, 20, 0],
    3: [35, 35, 35, 35, 34, 32, 30, 27, 25, 23, 20, 18, 14, 0],
    4: [35, 35, 33, 31, 29, 26, 24, 22, 20, 18, 15, 13, 9, 0],
    5: [32, 30, 28, 25, 23, 21, 19, 17, 15, 13, 10, 8, 4, 0],
    6: [25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 4, 2, 0, 0],
}


def _golden_window():
    out = {}
    for g, row in _GOLDEN_ROWS.items():
        for d, v in zip(range(-8, 6), row):
            if v is None:
                continue
            out[(g, d)] = v
    return out


def test_jump_table_matches_golden_table_exactly():
    window = build_jump_table().window
    assert window == _golden_window()


def test_jump_table_cache_returns_same_object():
    assert build_jump_table() is build_jump_table()


def _mirror_lands(physics, f, g, d):
    """Leftward mirror of the micro-level measurement."""
    R, C, rs = 48, 44, 30
    grid = [['.'] * C for _ in range(R)]
    for c in range(C - 20, C):                       # source ledge on the right
        grid[rs][c] = '#'
    t_left = C - 20 - g - 3                          # target ledge, to the left
    for c in range(t_left, t_left + 3):
        grid[rs - d][c] = '#'
    tiles = [''.join(row) for row in grid]
    level = Level.from_dict({
        'version': 3, 'tiles': tiles,
        'player_spawn': [C * TILE_SIZE - 88 - PLAYER_W, rs * TILE_SIZE - PLAYER_H],
        'gem': [0, 0], 'win_condition': 'gem',
    })
    game = Game(level, max_steps=None, physics=physics)
    game.player.grounded = True
    target_right = (t_left + 3) * TILE_SIZE
    for t in range(140):
        x = game.player.x
        mv = Action.LEFT if x + PLAYER_W > target_right - 2 else Action.NOOP
        action = (Action.LEFT_JUMP if mv is Action.LEFT else Action.JUMP) if t == f else mv
        game.step(action)
        if game.is_dead:
            return False
        if (game.player.grounded and game.player.y + PLAYER_H == (rs - d) * TILE_SIZE
                and game.player.x < target_right):
            return True
        if t > f >= 0 and game.player.grounded:
            return False
    return False


def test_jump_table_is_left_right_symmetric():
    right = build_jump_table()
    physics = PhysicsConfig()
    left = {}
    for g in range(G_MAX + 1):
        for d in range(-D_DOWN, D_UP + 1):
            if g == 0 and d == 0:
                continue
            landed = sum(_mirror_lands(physics, f, g, d) for f in range(-1, 46))
            left[(g, d)] = landed
    assert left == right.window


def test_weaker_jump_does_not_grow_windows():
    default = build_jump_table().window
    weaker = build_jump_table(PhysicsConfig(jump_velocity=-3.5)).window
    for key in default:
        assert weaker[key] <= default[key], f'{key}: {weaker[key]} > {default[key]}'


def test_walkoff_available_for_descends_not_climbs():
    # a plain walk-off lands on a ledge below (d < 0) but never on a higher one
    jt = build_jump_table()
    assert jt.walkoff[(1, -1)] is True
    assert jt.walkoff[(1, 1)] is False