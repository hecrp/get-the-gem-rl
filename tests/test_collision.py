from dataclasses import replace

import pytest

from game import Action, Game, Level
from game.collision import overlaps
from game.entities import Rect


def make_game(spawn=(16, 128), platforms=((0, 144, 240, 16),)):
    return Game(Level.from_dict(dict(width=240, height=160,
        player_spawn=spawn, gem=[216, 10], platforms=platforms)), max_steps=None)


@pytest.mark.parametrize('b,expected', [(Rect(12, 0, 8, 8), False),
    (Rect(0, 16, 8, 8), False), (Rect(11, 15, 8, 8), True),
    (Rect(-8, 0, 8, 8), False)])
def test_strict_aabb(b, expected):
    assert overlaps(Rect(0, 0, 12, 16), b) is expected


def test_landing_and_rest_without_vibration():
    game = make_game(spawn=(16, 50))
    for _ in range(50):
        state = game.step(Action.NOOP).state
    assert state.grounded and state.velocity_y == 0 and state.player_y == 128
    for _ in range(100):
        new = game.step(Action.NOOP).state
        assert replace(new, step_count=state.step_count) == state


@pytest.mark.parametrize('spawn,action,x', [((48, 128), Action.RIGHT, 48),
                                         ((72, 128), Action.LEFT, 72)])
def test_wall_both_directions(spawn, action, x):
    game = make_game(spawn, ((0, 144, 240, 16), (60, 100, 12, 44)))
    for _ in range(30):
        state = game.step(action).state
        assert state.player_x == x and state.velocity_x == 0
        assert state.player_y == 128


def test_jump_on_first_step_and_ceiling():
    game = make_game(platforms=((0, 144, 240, 16), (0, 100, 60, 8)))
    game.step(Action.NOOP)  # settle onto the floor: grounded becomes True (C-07)
    state = game.step(Action.JUMP).state
    assert state.velocity_y == -4.25 and state.player_y == 123.75
    assert not state.grounded
    for _ in range(6):
        state = game.step(Action.NOOP).state
        if state.velocity_y == 0:
            break
    assert state.player_y == 108 and state.velocity_y == 0 and not state.grounded


@pytest.mark.parametrize('spawn,action,x', [((0, 128), Action.LEFT, 0),
                                         ((228, 128), Action.RIGHT, 228)])
def test_world_side_bounds(spawn, action, x):
    state = make_game(spawn).step(action).state
    assert state.player_x == x and state.velocity_x == 0


def test_world_top_is_open():
    game = make_game((16, 0), ((0, 16, 240, 16),))
    game.step(Action.NOOP)  # settle onto the platform first
    state = game.step(Action.JUMP).state
    assert state.player_y == -4.25


def test_walking_off_edge_clears_grounded_without_coyote_jump():
    game = make_game((19, 128), ((0, 144, 20, 16),))
    state = game.step(Action.RIGHT).state
    assert not state.grounded
    state = game.step(Action.JUMP).state
    assert state.velocity_y == 0.5


@pytest.mark.parametrize('reverse', [False, True])
def test_simultaneous_overlapping_solids_resolve_nearest_face(reverse):
    # The lower top is encountered first in file order; both overlap after +.25.
    solids = [(0, 100.125, 40, 20), (10, 100, 40, 20)]
    if reverse:
        solids.reverse()
    game = make_game((16, 84), solids)
    for _ in range(10):
        state = game.step(Action.NOOP).state
        assert state.player_y == 84 and state.grounded
        assert not any(overlaps(game.player, p) for p in game.level.platforms)


def test_jump_lands_then_held_jump_repeats():
    game = make_game()
    game.step(Action.NOOP)  # settle onto the floor: grounded becomes True (C-07)
    first = game.step(Action.JUMP).state
    assert first.velocity_y == -4.25
    saw_landing = False
    for _ in range(100):
        state = game.step(Action.JUMP).state
        if saw_landing:
            assert state.velocity_y == -4.25
            return
        saw_landing = state.grounded
    pytest.fail('never landed')
