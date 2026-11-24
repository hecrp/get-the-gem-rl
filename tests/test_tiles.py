"""Tiles format (version 3) parsing, fusion and equivalence."""
import json

import pytest

from game import Action, Game, Level
from game.entities import Exit, Gem, Platform


def grid(placements=None, W=30, H=20):
    """Build a WxH all-dot tile map; placements maps (row, col) -> char."""
    rows = [['.'] * W for _ in range(H)]
    for (r, c), ch in (placements or {}).items():
        rows[r][c] = ch
    return [''.join(r) for r in rows]


def tiles_level(tiles, **extra):
    data = {"version": 3, "tile_size": 8, "tiles": tiles}
    data.update(extra)
    return Level.from_dict(data)


def floor_row(H=20, r=19):
    return {(r, c): '#' for c in range(30)}


def test_level_007_parses_to_known_geometry():
    level = Level.from_json("levels/level_007.json")
    assert level.width == 480 and level.height == 320
    assert level.version == 3 and level.name == "First big one"
    assert level.max_steps == 3600
    assert level.win_condition == "exit_with_gem"
    assert level.player_spawn == (32.0, 288.0)
    assert level.gem == Gem(56.0, 72.0, 8.0, 8.0)
    assert level.exit == Exit(8.0, 288.0, 12.0, 16.0)
    assert len(level.platforms) == 9
    assert len(level.hazards) == 2
    assert level.tiles is not None and len(level.tiles) == 40


def test_solid_fusion_horizontal_and_vertical():
    placements = {(5, c): '#' for c in range(3)} | {(6, c): '#' for c in range(3)}
    level = tiles_level(grid(placements), player_spawn=[40, 0], gem=[200, 0])
    assert level.platforms == (Platform(0.0, 40.0, 24.0, 16.0),)


def test_spike_fusion_horizontal_only():
    placements = {(5, c): '^' for c in range(3, 6)}
    level = tiles_level(grid(placements), player_spawn=[40, 0], gem=[200, 0])
    assert [(h.x, h.y, h.w, h.h) for h in level.hazards] == [(24.0, 44.0, 24.0, 4.0)]


def test_dimensions_derived_and_mismatch_rejected():
    tiles = grid()
    assert tiles_level(tiles, player_spawn=[16, 0], gem=[200, 0]).width == 240
    assert tiles_level(tiles, player_spawn=[16, 0], gem=[200, 0]).height == 160
    with pytest.raises(ValueError, match='width'):
        tiles_level(tiles, player_spawn=[16, 0], gem=[200, 0], width=241)


def test_unknown_character_reports_row_and_column():
    tiles = grid({(1, 3): 'X'})
    with pytest.raises(ValueError, match=r'X.*row 1, column 3'):
        tiles_level(tiles, player_spawn=[16, 0], gem=[200, 0])


def test_ragged_rows_rejected():
    rows = grid()
    rows[1] = rows[1][:-1]
    with pytest.raises(ValueError, match='row 1 has length'):
        tiles_level(rows, player_spawn=[16, 0], gem=[200, 0])


def test_p_and_g_required_exactly_once_each_place():
    # missing P (tiles have G only, no explicit spawn)
    with pytest.raises(ValueError, match='player_spawn'):
        tiles_level(grid({(10, 5): 'G'}), gem=[40, 80])
    # P in both tiles and explicit
    with pytest.raises(ValueError, match='both tiles and explicit'):
        tiles_level(grid({(18, 5): 'P', (10, 5): 'G'}), player_spawn=[40, 136])
    # duplicate G
    with pytest.raises(ValueError, match='multiple G'):
        tiles_level(grid({(18, 5): 'P', (10, 2): 'G', (10, 8): 'G'}))
    # duplicate D
    with pytest.raises(ValueError, match='multiple D'):
        tiles_level(grid({(18, 5): 'P', (10, 5): 'G', (18, 2): 'D', (18, 8): 'D'}))


def test_tiles_coexist_with_explicit_platforms():
    placements = floor_row() | {(18, 5): 'P', (10, 5): 'G'}
    level = tiles_level(grid(placements), platforms=[[200, 0, 8, 8]])
    # tiles floor first, then the explicit platform
    assert level.platforms[0] == Platform(0.0, 152.0, 240.0, 8.0)
    assert level.platforms[-1] == Platform(200.0, 0.0, 8.0, 8.0)


def test_tiles_and_rectangles_give_same_trajectory():
    placements = floor_row() | {(18, 5): 'P', (10, 5): 'G'}
    placements |= {(12, c): '#' for c in range(10, 15)}
    as_tiles = tiles_level(grid(placements))
    as_rects = Level.from_dict({
        'version': 1, 'width': 240, 'height': 160,
        'player_spawn': [40, 136], 'gem': [40, 80],
        'platforms': [[0, 152, 240, 8], [80, 96, 40, 8]],
    })
    assert set(as_tiles.platforms) == set(as_rects.platforms)
    actions = [Action.RIGHT, Action.RIGHT, Action.JUMP, Action.NOOP,
               Action.RIGHT, Action.LEFT, Action.LEFT_JUMP]
    g1, g2 = Game(as_tiles), Game(as_rects)
    for a in actions:
        s1, s2 = g1.step(a).state, g2.step(a).state
        assert s1 == s2


def test_version_three_accepted_by_loader():
    level = tiles_level(grid({(18, 5): 'P', (10, 5): 'G'}), version=3)
    assert level.version == 3
