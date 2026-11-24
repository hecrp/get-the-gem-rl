from dataclasses import FrozenInstanceError
import json
import random

import pytest


def make_level(**updates):
    from game import Level
    data = dict(width=240, height=160, player_spawn=[16, 128],
                gem=[216, 132], platforms=[[0, 144, 240, 16]])
    data.update(updates)
    return Level.from_dict(data)


def test_constructor_snapshot_reset_rng_and_serialization():
    from game import Game, GameState, StepResult, Action
    game = Game(make_level())
    state = game.get_state()
    assert isinstance(state, GameState)
    assert state == game.reset()
    assert state is not game.get_state()
    assert hash(state) == hash(game.get_state())
    assert (state.player_x, state.player_y, state.velocity_x, state.velocity_y) == (16, 128, 0, 0)
    # A spawn resting on a platform is grounded at reset.
    assert state.grounded and state.alive and not state.gem_collected
    assert not any((game.is_done, game.is_won, game.is_dead, game.is_timeout))
    assert game.step_count == 0
    with pytest.raises(FrozenInstanceError):
        state.player_x = 0
    serialized = game.get_state_dict()
    assert json.loads(json.dumps(serialized)) == serialized
    serialized['player']['x'] = -1
    assert game.get_state() == state
    expected = random.Random(0)
    assert game.rng.random() == expected.random()
    game.reset()
    assert game.rng.random() == expected.random()
    game.reset(seed=9)
    assert game.rng.random() == random.Random(9).random()
    assert len(Action) == 6
    assert [a.value for a in Action] == list(range(6))


@pytest.mark.parametrize('value', [0, -1, True, 1.5, '1'])
def test_invalid_max_steps(value):
    from game import Game
    with pytest.raises(ValueError, match='max_steps'):
        Game(make_level(), max_steps=value)
