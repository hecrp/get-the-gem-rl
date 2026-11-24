"""Level 7 conformity (exact numbers) and replay of every reference solution.

reset() probes the ground, so a JUMP issued on the very first step takes off.
The reference solutions were produced with tools/solve_level.py (blind: these
levels carry no waypoints) and the numbers below measured from their replays.
"""
import json
from pathlib import Path

import pytest

from game import Action, Game, Level

ROOT = Path(__file__).resolve().parents[1]


def replay_rle(level_name):
    sol = json.loads((ROOT / 'levels' / 'solutions' / level_name).read_text())
    actions = []
    for a, rep in sol['actions']:
        actions.extend([a] * rep)
    game = Game.from_level(ROOT / 'levels' / level_name, max_steps=None, seed=0)
    gem_at = win_at = None
    for i, a in enumerate(actions, 1):
        r = game.step(a)
        if gem_at is None and r.state.gem_collected:
            gem_at = i
        if r.won:
            win_at = i
            break
    return sol, gem_at, win_at, game.is_won, game.is_exit_open


def test_level_007_conforms_exactly():
    sol, gem_at, win_at, won, _ = replay_rle('level_007.json')
    # The reference solution takes 766 steps and collects the gem at 689.
    assert gem_at == 689 and win_at == 766 and won
    assert sol['total_steps'] == 766 and sol['gem_collected_at_step'] == 689


def test_level_007_single_action_outcomes():
    # RIGHT -> dead at 101 (pit)
    g = Game.from_level(ROOT / 'levels' / 'level_007.json', max_steps=None, seed=0)
    for i in range(1, 1000):
        if g.step(Action.RIGHT).dead:
            assert i == 101
            break
    # RIGHT_JUMP -> dead at 143 (spikes): the first-step jump takes off
    g = Game.from_level(ROOT / 'levels' / 'level_007.json', max_steps=None, seed=0)
    for i in range(1, 1000):
        if g.step(Action.RIGHT_JUMP).dead:
            assert i == 143
            break
    # LEFT -> timeout (level max_steps=3600)
    g = Game.from_level(ROOT / 'levels' / 'level_007.json', seed=0)
    assert g.max_steps == 3600
    for i in range(1, 4000):
        if g.step(Action.LEFT).timeout:
            assert i == 3600
            break
    else:
        raise AssertionError('LEFT never timed out')


@pytest.mark.parametrize('number', [1, 2, 3, 4, 5, 6])
def test_levels_1_to_6_replay_and_max_steps(number):
    # Blind-solver reference solutions for the single-screen levels; every
    # one must win, and the exit levels must open
    # the door. The max_steps >= 2x rule uses the effective cap (levels 1-6
    # have no max_steps field, so Game applies DEFAULT_MAX_STEPS = 1800).
    from game.constants import DEFAULT_MAX_STEPS
    name = f'level_{number:03}.json'
    sol, gem_at, win_at, won, exit_open = replay_rle(name)
    assert won and gem_at is not None
    if number >= 4:
        assert exit_open
    level = Level.from_json(ROOT / 'levels' / name)
    effective = level.max_steps if level.max_steps is not None else DEFAULT_MAX_STEPS
    assert effective >= 2 * sol['total_steps']


@pytest.mark.parametrize('number', [7, 8, 9])
def test_levels_7_to_9_max_steps_rule(number):
    name = f'level_{number:03}.json'
    sol, gem_at, win_at, won, _ = replay_rle(name)
    assert won
    level = Level.from_json(ROOT / 'levels' / name)
    assert level.max_steps >= 2 * sol['total_steps']


@pytest.mark.parametrize('number', [8, 9])
def test_levels_8_and_9_replay_and_max_steps(number):
    name = f'level_{number:03}.json'
    sol, gem_at, win_at, won, _ = replay_rle(name)
    assert won and gem_at is not None
    assert sol['total_steps'] >= 1
    level = Level.from_json(ROOT / 'levels' / name)
    # max_steps >= 2x the reference solution
    assert level.max_steps >= 2 * sol['total_steps']
