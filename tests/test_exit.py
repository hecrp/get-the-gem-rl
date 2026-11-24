"""Gem -> door mechanics, reference solutions and backward compatibility."""
import json
from pathlib import Path

import pytest

from game import Action, Game, GameState, Level

ROOT = Path(__file__).resolve().parents[1]
LEVELS = tuple(ROOT / 'levels' / f'level_{i:03}.json' for i in range(1, 7))


def level_from(**updates):
    data = dict(width=240, height=160, player_spawn=[16, 128],
                gem=[216, 132], platforms=[[0, 144, 240, 16]])
    data.update(updates)
    return Game(Level.from_dict(data))


def test_touching_closed_door_without_gem_does_nothing():
    game = level_from(exit=[16, 128])  # spawn overlaps the door; valid metroidvania design
    assert game.level.win_condition == 'exit_with_gem'
    assert not game.is_done and not game.is_exit_open
    for _ in range(30):
        result = game.step(Action.RIGHT)
        assert not result.done and not result.won and not result.state.exit_open
        assert not game.has_gem


def test_collect_continues_then_door_wins():
    game = Game.from_level(LEVELS[3])  # level_004
    seen_collected = False
    for _ in range(300):
        result = game.step(Action.RIGHT if not game.has_gem else Action.LEFT)
        if game.has_gem and not seen_collected:
            seen_collected = True
            assert not result.done and not result.won  # collecting does not end
        if result.done:
            break
    assert game.is_won and game.is_exit_open and game.has_gem
    assert result.done and result.won


def test_collection_is_permanent_and_door_is_only_win_path():
    game = Game.from_level(LEVELS[3])
    while not game.has_gem:
        game.step(Action.RIGHT)
    assert not game.is_won and not game.is_done
    for _ in range(40):  # wander back over the gem spawn region without returning to the door
        result = game.step(Action.LEFT if game.step_count % 6 == 0 else Action.RIGHT)
        assert game.has_gem  # never reverts
        assert not result.won or game.is_exit_open  # only the door can win here
    assert not game.is_exit_open


def test_death_with_gem_keeps_collected_flag():
    game = level_from(gem=[40, 128], exit=[200, 128],
                      platforms=[[0, 144, 80, 16], [140, 144, 60, 16]])
    while not game.is_done:
        result = game.step(Action.RIGHT)
    assert result.dead and not result.won and not result.timeout
    assert game.has_gem and game.get_state().gem_collected
    assert result.dead == (not result.state.alive)


def test_death_wins_over_collection_in_same_step():
    game = level_from(gem=[28, 132], hazards=[[28, 132, 8, 8]],
                      exit=[200, 128], win_condition='exit_with_gem')
    result = game.step(Action.RIGHT)
    assert result.dead and not result.won and not result.timeout
    assert not game.has_gem  # death is evaluated before collection


def test_reset_restores_gem_and_door_after_collection_and_win():
    game = Game.from_level(LEVELS[3])
    initial = game.get_state()
    assert not initial.gem_collected and not initial.exit_open
    while not game.has_gem:
        game.step(Action.RIGHT)
    assert game.get_state() != initial
    assert game.reset() == initial
    assert not game.has_gem and not game.is_exit_open
    while not game.is_done:
        game.step(Action.RIGHT if not game.has_gem else Action.LEFT)
    assert game.is_won and game.is_exit_open
    assert game.reset() == initial
    assert not game.has_gem and not game.is_exit_open


def test_timeout_with_collected_gem_is_not_a_win():
    game = Game.from_level(LEVELS[3], max_steps=150)
    while not game.is_done:
        game.step(Action.RIGHT)
    assert game.is_timeout and game.has_gem and not game.is_won
    assert not game.is_exit_open


def test_decorative_door_wins_by_gem_and_never_opens():
    game = level_from(exit=[24, 128], win_condition='gem')
    result = None
    while not game.is_done:
        result = game.step(Action.RIGHT)
    assert result.won and not result.dead and not result.timeout
    assert game.has_gem and not game.is_exit_open
    assert game.level.win_condition == 'gem'


def test_version1_with_exit_defaults_to_exit_with_gem():
    game = Game(Level.from_dict(dict(width=240, height=160, version=1,
                                     player_spawn=[60, 128], gem=[216, 132],
                                     exit=[16, 128], platforms=[[0, 144, 240, 16]])))
    assert game.level.win_condition == 'exit_with_gem'
    while not game.has_gem:
        result = game.step(Action.RIGHT)
    assert not result.done  # collecting no longer wins for a level with a door
    assert not game.is_won


def test_action_enum_unchanged_and_state_fields_appended():
    assert len(Action) == 6
    assert [a.value for a in Action] == list(range(6))
    state = Game(Level.from_dict(dict(width=240, height=160, player_spawn=[16, 128],
                                      gem=[216, 132], platforms=[[0, 144, 240, 16]]))).get_state()
    assert isinstance(state, GameState)
    assert (state.exit_x, state.exit_y, state.exit_open) == (None, None, False)
    assert hash(state) is not None


def test_state_dict_serializes_with_and_without_door():
    with_door = Game.from_level(LEVELS[3]).get_state_dict()
    without_door = Game.from_level(LEVELS[0]).get_state_dict()
    assert json.loads(json.dumps(with_door)) == with_door
    assert json.loads(json.dumps(without_door)) == without_door
    assert with_door['win_condition'] == 'exit_with_gem'
    assert without_door['win_condition'] == 'gem'
    assert with_door['exit'] == {'x': 16.0, 'y': 128.0, 'open': False}
    assert without_door['exit'] is None


@pytest.mark.parametrize('number', [4, 5, 6])
def test_scripted_solution_reaches_door_with_gem(number):
    game = Game.from_level(LEVELS[number - 1], seed=1)
    outbound = {
        5: ((30, 40), (92, 100), (160, 168)),
        6: ((70, 78), (128, 136), (176, 186)),
    }.get(number, ())
    for _ in range(500):
        if game.is_done:
            break
        state = game.get_state()
        if not game.has_gem:
            jump = state.grounded and any(lo <= state.player_x <= hi for lo, hi in outbound)
            action = Action.RIGHT_JUMP if jump else Action.RIGHT
        else:
            action = Action.LEFT
            if number == 6 and state.grounded:
                if state.player_y <= 100.5:
                    if state.player_x <= 198:
                        action = Action.LEFT_JUMP
                elif any(lo <= state.player_x <= hi for lo, hi in ((176, 190), (126, 138))):
                    action = Action.LEFT_JUMP
        game.step(action)
    assert game.is_won and game.is_exit_open and game.has_gem


def test_reference_negative_cases_level4_and_level6():
    right = Game.from_level(LEVELS[3])
    while not right.is_done:
        right.step(Action.RIGHT)
    assert right.is_timeout and right.has_gem
    left = Game.from_level(LEVELS[3])
    while not left.is_done:
        left.step(Action.LEFT)
    assert left.is_timeout and not left.has_gem
    jump = Game.from_level(LEVELS[5])
    while not jump.is_done:
        jump.step(Action.RIGHT_JUMP)
    assert jump.is_dead and not jump.has_gem