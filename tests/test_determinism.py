"""Independent engine acceptance tests: exact replay and 100,000 live steps."""
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import pytest

from game import Action, Game

ROOT = Path(__file__).resolve().parents[1]
LEVELS = tuple(ROOT / "levels" / f"level_{i:03}.json" for i in range(1, 10))


def test_core_runs_without_site_packages_or_graphics():
    """-S proves the simulation needs only the standard library."""
    script = """
import json, sys
from game import Game, Action
assert 'pygame' not in sys.modules
assert 'numpy' not in sys.modules
game = Game.from_level('levels/level_001.json')
while not game.is_done:
    result = game.step(Action.RIGHT)
assert result.won
print(json.dumps(game.get_state_dict()))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], cwd=ROOT,
        capture_output=True, text=True, check=True,
    )
    assert isinstance(json.loads(result.stdout), dict)


@pytest.mark.parametrize("level_path", LEVELS, ids=lambda p: p.stem)
def test_same_seed_actions_and_resets_match_every_snapshot(level_path):
    first = Game.from_level(level_path, seed=42, max_steps=250)
    second = Game.from_level(level_path, seed=42, max_steps=250)
    rng = random.Random(9876)
    initial = first.get_state()
    assert initial == second.get_state()
    assert hash(initial) == hash(second.get_state())
    for _ in range(2000):
        if first.is_done:
            assert first.reset() == second.reset() == initial
        action = rng.choice(tuple(Action))
        assert first.step(action) == second.step(int(action))
        assert first.get_state_dict() == second.get_state_dict()
    assert initial.step_count == 0  # older snapshots are not mutated


def test_game_rng_is_local_and_reset_does_not_reseed_unless_requested():
    global_before = random.getstate()
    first = Game.from_level(LEVELS[0])
    second = Game.from_level(LEVELS[0])
    assert first.rng.random() == second.rng.random()
    previous = first.rng.getstate()
    first.reset()
    assert first.rng.getstate() == previous
    first.reset(seed=81)
    expected = random.Random(81)
    assert first.rng.random() == expected.random()
    assert random.getstate() == global_before


def overlaps(x, y, w, h, rect):
    # Deliberately independent of the implementation's collision helper.
    return x < rect.x + rect.w and x + w > rect.x and y < rect.y + rect.h and y + h > rect.y


def test_100000_seeded_actions_preserve_invariants_across_all_levels():
    rng = random.Random(1234)
    actions = tuple(Action)
    executed = 0
    reset_count = 0
    for index, level_path in enumerate(LEVELS):
        game = Game.from_level(level_path, seed=1234, max_steps=500)
        initial = game.get_state()
        action = Action.NOOP
        remaining = 0
        budget = 100_000 // len(LEVELS) + (index < 100_000 % len(LEVELS))
        for _ in range(budget):
            if game.is_done:
                terminal = game.step(Action.RIGHT_JUMP)
                assert game.step(Action.LEFT) == terminal
                assert game.reset() == initial
                reset_count += 1
                remaining = 0
            if remaining == 0:
                action = rng.choice(actions)
                remaining = rng.randint(1, 20)
            remaining -= 1
            before = game.step_count
            result = game.step(action)
            state = result.state
            executed += 1
            assert state.step_count == before + 1
            assert state == game.get_state()
            x, y, vx, vy = state.player_x, state.player_y, state.velocity_x, state.velocity_y
            assert all(math.isfinite(v) for v in (x, y, vx, vy))
            assert 0 <= x <= game.level.width - 12
            assert y <= game.level.height + 16 + 4.0
            assert abs(vx) <= 1.5
            assert -4.5 <= vy <= 4.0
            if state.alive:
                assert not any(overlaps(x, y, 12, 16, platform) for platform in game.level.platforms)
            assert result.done == (result.won or result.dead or result.timeout)
            assert sum((result.won, result.dead, result.timeout)) <= 1
            assert result.dead == (not state.alive) == game.is_dead
            assert result.won == state.won == game.is_won
            assert state.gem_collected == game.has_gem
            assert (not result.won) or state.gem_collected  # winning requires the gem
            assert result.timeout == state.timeout == game.is_timeout
            assert result.done == game.is_done
            assert not state.grounded or vy == 0
            assert state.step_count <= 500
    assert executed == 100_000
    assert reset_count > 0
