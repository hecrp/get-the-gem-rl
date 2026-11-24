"""Broad-phase grid: superset guarantee + bit-identical to brute force."""
import random

from game import Game, Level
from game.collision import overlaps


def test_solids_near_is_ordered_superset_of_overlaps():
    level = Level.from_json('levels/level_007.json')
    rng = random.Random(5)
    for _ in range(2000):
        x = rng.uniform(-20, 520)
        y = rng.uniform(-20, 360)
        w = rng.uniform(4, 40)
        h = rng.uniform(4, 40)
        indices = level.solids_near(x, y, w, h)
        # ascending, no duplicates
        assert tuple(indices) == tuple(sorted(set(indices)))
        from game.entities import Rect
        probe = Rect(x, y, w, h)
        truly_overlapping = {i for i, p in enumerate(level.platforms)
                             if overlaps(probe, p)}
        assert truly_overlapping <= set(indices)


def test_grid_and_bruteforce_trajectories_identical_on_level_7():
    level = Level.from_json('levels/level_007.json')
    rng = random.Random(999)
    actions = [rng.randrange(6) for _ in range(50_000)]

    # grid version
    grid_game = Game(level, seed=0)
    grid_states = []
    for a in actions:
        grid_states.append(grid_game.step(a).state)

    # brute-force version: make solids_near return every platform index
    original = Level.solids_near
    Level.solids_near = lambda self, x, y, w, h: tuple(range(len(self.platforms)))
    try:
        brute_game = Game(level, seed=0)
        for i, a in enumerate(actions):
            assert brute_game.step(a).state == grid_states[i]
    finally:
        Level.solids_near = original
