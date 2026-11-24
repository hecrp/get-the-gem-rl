"""tools/solve_level.py: solves known levels, detects impossible ones."""
import json
from pathlib import Path

import pytest

from tools.solve_level import solve

ROOT = Path(__file__).resolve().parents[1]


def replay_wins(level_name, solution):
    from game import Game
    actions = []
    for a, rep in solution['actions']:
        actions.extend([a] * rep)
    game = Game.from_level(ROOT / 'levels' / level_name, max_steps=None, seed=0)
    for a in actions:
        if game.step(a).won:
            return True
    return game.is_won


@pytest.mark.parametrize('number', [1, 4, 7])
def test_solver_finds_a_winning_solution(number):
    level = ROOT / 'levels' / f'level_{number:03}.json'
    sequence, nodes = solve(level)
    assert sequence is not None and nodes > 0
    # replay the RLE-compressed solution
    from tools.solve_level import _rle, _metrics
    total, gem_at = _metrics(level, sequence)
    assert total == len(sequence)
    game = Game_replay(level, sequence)
    assert game.is_won


def Game_replay(level, sequence):
    from game import Game
    game = Game.from_level(level, max_steps=None, seed=0)
    for a in sequence:
        game.step(a)
    return game


def test_solver_is_deterministic():
    level = ROOT / 'levels' / 'level_001.json'
    a, _ = solve(level)
    b, _ = solve(level)
    assert a == b


def test_solver_reports_no_solution_for_enclosed_gem(tmp_path):
    # gem sealed inside a solid box, unreachable from the spawn
    data = {
        'version': 1, 'width': 240, 'height': 160,
        'player_spawn': [16, 128], 'gem': [120, 40],
        'platforms': [
            [0, 144, 240, 16],     # floor
            [100, 32, 40, 8],      # box around the gem
            [100, 40, 8, 48],
            [132, 40, 8, 48],
            [100, 88, 40, 8],
        ],
    }
    path = tmp_path / 'impossible.json'
    path.write_text(json.dumps(data))
    sequence, nodes = solve(path, limit=20000)
    assert sequence is None
