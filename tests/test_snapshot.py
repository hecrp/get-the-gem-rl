"""save_state / load_state contract."""
import random

import pytest

from game import Action, Game, Level


def make_level(width=240, height=160, win_condition=None):
    data = dict(width=width, height=height, player_spawn=[16, 128],
                gem=[216, 132], platforms=[[0, 144, width, 16]])
    if win_condition:
        data['win_condition'] = win_condition
        data['exit'] = [8, 128]
    return Level.from_dict(data)


def test_save_load_reproduces_same_future_states():
    level = make_level()
    game = Game(level, seed=42)
    rng = random.Random(123)
    game.step(Action.RIGHT)
    game.step(Action.JUMP)
    snap = game.save_state()

    # advance further on a copy
    game2 = Game(level, seed=42)
    game2.load_state(snap)
    game3 = Game(level, seed=42)
    game3.load_state(snap)

    actions = [rng.randrange(6) for _ in range(200)]
    for a in actions:
        assert game2.step(a).state == game3.step(a).state


def test_snapshot_restores_gem_collected_and_rng():
    level = make_level(win_condition='exit_with_gem')
    game = Game(level, seed=7)
    # walk right to collect the gem
    for _ in range(300):
        r = game.step(Action.RIGHT)
        if r.state.gem_collected:
            break
    assert game.has_gem
    snap = game.save_state()

    game2 = Game(level, seed=7)
    game2.load_state(snap)
    assert game2.has_gem
    assert game2.player.x == game.player.x
    assert game2.rng.getstate() == game.rng.getstate()


def test_snapshot_of_terminated_episode_restores_terminal_flags():
    level = make_level(win_condition='exit_with_gem')
    game = Game(level, max_steps=None)
    # collect gem then walk left to the door
    while not game.has_gem:
        game.step(Action.RIGHT)
    while not game.is_done:
        game.step(Action.LEFT)
    assert game.is_won
    snap = game.save_state()
    game2 = Game(level, max_steps=None)
    game2.load_state(snap)
    assert game2.is_won and game2.is_done


def test_load_state_rejects_different_level():
    game = Game(make_level())
    snap = game.save_state()
    other = Game(make_level(width=320))
    with pytest.raises(ValueError, match='level'):
        other.load_state(snap)


def test_snapshot_round_trip_is_cheap_and_reversible():
    level = make_level()
    game = Game(level, seed=1)
    for a in (Action.RIGHT, Action.RIGHT_JUMP, Action.NOOP):
        game.step(a)
    snap = game.save_state()
    before = game.get_state()
    for _ in range(50):
        game.step(Action.LEFT)
    game.load_state(snap)
    assert game.get_state() == before
