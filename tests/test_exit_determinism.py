"""Door levels: six-level invariants and replay after real gem collection.

No internal
flags or player coordinates are injected to obtain a carrying state.
"""
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest

from game import Action, Game

ROOT = Path(__file__).resolve().parents[1]
LEVELS = tuple(ROOT / 'levels' / f'level_{i:03}.json' for i in range(1, 7))


def scripted_outbound_action(game, number):
    state = game.get_state()
    intervals = {4: (), 5: ((30, 40), (92, 100), (160, 168)),
                 6: ((70, 78), (128, 136), (176, 186))}[number]
    jump = state.grounded and any(lo <= state.player_x <= hi for lo, hi in intervals)
    return Action.RIGHT_JUMP if jump else Action.RIGHT


def assert_invariants(game, result, previous):
    state = result.state
    x, y, vx, vy = state.player_x, state.player_y, state.velocity_x, state.velocity_y
    assert all(math.isfinite(v) for v in (x, y, vx, vy))
    assert 0 <= x <= game.level.width - 12
    assert y <= game.level.height + 16 + 4.0
    assert abs(vx) <= 1.5
    assert -4.5 <= vy <= 4.0
    if state.alive:
        for solid in game.level.platforms:
            assert not (x < solid.x + solid.w and x + 12 > solid.x
                        and y < solid.y + solid.h and y + 16 > solid.y)
    assert result.done == (result.won or result.dead or result.timeout) == game.is_done
    assert sum((result.won, result.dead, result.timeout)) <= 1
    assert result.dead == (not state.alive) == game.is_dead
    assert result.won == state.won == game.is_won
    assert result.timeout == state.timeout == game.is_timeout
    assert not state.won or state.gem_collected
    assert not state.exit_open or (state.won and state.gem_collected)
    assert not previous.gem_collected or state.gem_collected
    assert state.gem_collected == game.has_gem
    assert state.exit_open == game.is_exit_open
    if game.level.win_condition == 'gem':
        assert state.won == state.gem_collected
        assert not state.exit_open
    else:
        assert state.won == state.exit_open
    assert not state.grounded or vy == 0
    assert state.step_count == previous.step_count + 1
    assert game.max_steps is None or state.step_count <= game.max_steps
    assert state == game.get_state()
    assert (state.gem_x, state.gem_y) == (game.level.gem.x, game.level.gem.y)
    if game.level.exit is None:
        assert (state.exit_x, state.exit_y) == (None, None)
    else:
        assert (state.exit_x, state.exit_y) == (game.level.exit.x, game.level.exit.y)


def test_100000_live_steps_across_six_levels_with_door_invariants():
    rng = random.Random(1234)
    actions = tuple(Action)
    executed = 0
    resets = 0
    for index, path in enumerate(LEVELS):
        game = Game.from_level(path, seed=1234, max_steps=500)
        initial = game.get_state()
        action = Action.NOOP
        held = 0
        budget = 100_000 // len(LEVELS) + (index < 100_000 % len(LEVELS))
        for _ in range(budget):
            if game.is_done:
                terminal = game.step(Action.NOOP)
                assert game.step(Action.RIGHT_JUMP) == terminal
                assert game.reset() == initial
                held = 0
                resets += 1
            if held == 0:
                action = rng.choice(actions)
                held = rng.randint(1, 20)
            held -= 1
            previous = game.get_state()
            assert_invariants(game, game.step(action), previous)
            executed += 1
    assert executed == 100_000
    assert resets > 0


@pytest.mark.parametrize('number', [4, 5, 6])
def test_seeded_replay_and_random_continuation_after_scripted_collection(number):
    rng = random.Random(4100 + number)
    actions = tuple(Action)
    first = Game.from_level(LEVELS[number - 1], seed=81, max_steps=350)
    second = Game.from_level(LEVELS[number - 1], seed=81, max_steps=350)
    initial = first.get_state()
    continued_with_gem = 0
    for _ in range(10):
        assert first.reset() == second.reset() == initial
        # Reach the gem exclusively through public simulation actions.
        for _ in range(250):
            previous = first.get_state()
            action = scripted_outbound_action(first, number)
            result = first.step(action)
            assert result == second.step(int(action))
            assert_invariants(first, result, previous)
            if first.has_gem:
                break
            assert not first.is_done
        assert first.has_gem and not first.is_done and not first.is_exit_open
        assert hash(first.get_state()) == hash(second.get_state())
        held = 0
        for _ in range(250):
            if first.is_done:
                break
            if held == 0:
                action = rng.choice(actions)
                held = rng.randint(1, 20)
            held -= 1
            previous = first.get_state()
            result = first.step(action)
            assert result == second.step(int(action))
            assert_invariants(first, result, previous)
            assert first.has_gem
            continued_with_gem += 1
        assert first.get_state_dict() == second.get_state_dict()
        assert json.loads(json.dumps(first.get_state_dict())) == first.get_state_dict()
    assert continued_with_gem > 0
    assert first.reset() == second.reset() == initial


def test_door_level_can_win_and_serialize_without_site_packages():
    script = """
import json, sys
from game import Game, Action
world = Game.from_level('levels/level_004.json')
while not world.is_done:
    world.step(Action.LEFT if world.has_gem else Action.RIGHT)
assert world.is_won and world.is_exit_open
assert 'numpy' not in sys.modules and 'pygame' not in sys.modules
assert len(Action) == 6
print(json.dumps(world.get_state_dict()))
"""
    result = subprocess.run([sys.executable, '-S', '-c', script], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    state = json.loads(result.stdout)
    assert state['win_condition'] == 'exit_with_gem'
    assert state['exit']['open']
    assert state['gem']['collected']
