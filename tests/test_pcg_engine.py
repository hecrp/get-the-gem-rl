"""Engine integration: Game.reset(level=...) and solvability of generated
levels, zero soft-locks."""
import random

import pytest

from game import Game
from pcg import PRESETS, GenConfig, LevelDistribution, generate_level


@pytest.fixture(scope='module')
def two_levels():
    return generate_level(0, PRESETS['tiny']), generate_level(1, PRESETS['small'])


def test_reset_swaps_level_and_restores_state(two_levels):
    lv0, lv1 = two_levels
    game = Game(lv0, seed=123)
    for _ in range(10):
        game.step(2)
    game.reset(level=lv1)
    assert game.level is lv1
    assert game.player.x == lv1.player_spawn[0]
    assert game.player.y == lv1.player_spawn[1]
    assert game.step_count == 0
    assert not game.is_won and not game.is_dead and not game.is_timeout


def test_reset_reresolves_max_steps_from_level(two_levels):
    lv0, lv1 = two_levels
    game = Game(lv0)                       # default FROM_LEVEL
    assert game.max_steps == lv0.max_steps
    game.reset(level=lv1)
    assert game.max_steps == lv1.max_steps


def test_reset_keeps_explicit_max_steps(two_levels):
    lv0, lv1 = two_levels
    game = Game(lv0, max_steps=500)
    game.reset(level=lv1)
    assert game.max_steps == 500


def test_reset_invalidates_snapshots_from_other_level(two_levels):
    lv0, lv1 = two_levels
    game = Game(lv0)
    snap = game.save_state()
    game.reset(level=lv1)
    with pytest.raises(ValueError):
        game.load_state(snap)


def test_alternating_level_sequence_is_deterministic():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=0, num_levels=10)

    def run():
        game = Game(dist.get(0), seed=42)
        states = []
        for i in range(5):
            game.reset(level=dist.get(i))
            for a in (2, 2, 5, 2, 2, 2):
                game.step(a)
            s = game.get_state()
            states.append((s.player_x, s.player_y, s.velocity_y, s.grounded, s.won))
        return states

    assert run() == run()


# ---- solvability (slow) ------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('preset', ['tiny', 'small', 'wide', 'tall'])
@pytest.mark.parametrize('difficulty', [0.0, 0.5, 1.0])
def test_generated_levels_are_solvable(preset, difficulty):
    from tools.solve_level import solve
    cfg = GenConfig(cols=PRESETS[preset].cols, rows=PRESETS[preset].rows,
                    difficulty=difficulty)
    for seed in range(100):
        level = generate_level(seed, cfg)
        sequence, _ = solve(level, k=3)
        assert sequence is not None, f'seed {seed} ({preset}, d={difficulty}) unsolvable'


@pytest.mark.slow
def test_large_and_huge_levels_are_solvable():
    from tools.solve_level import solve
    for seed in range(3):
        level = generate_level(seed, PRESETS['large'])
        assert solve(level, k=3)[0] is not None, f'large seed {seed}'
    level = generate_level(0, PRESETS['huge'])
    assert solve(level, k=3)[0] is not None, 'huge seed 0'


@pytest.mark.slow
def test_no_softlock_after_random_actions():
    from tools.solve_level import solve
    for seed in range(20):
        level = generate_level(seed, PRESETS['small'])
        game = Game(level, seed=seed)
        rng = random.Random(seed + 1000)
        for _ in range(200):
            if game.step(rng.randrange(6)).done:
                break
        if not game.is_done:
            sequence, _ = solve(level, k=3, start_state=game.save_state())
            assert sequence is not None, f'seed {seed}: soft-lock after random actions'