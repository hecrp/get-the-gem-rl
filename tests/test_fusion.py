"""Fusion vs no-fusion: same map, identical trajectories."""
import random

from game import Action, Game, Level
from game.entities import Platform

# 30x20 tile map with a floor and a 3x2 solid block (the block fuses to one rect).
TILES = [
    "." * 30, "." * 30, "." * 30, "." * 30, "." * 30,
    "." * 30, "." * 30, "." * 30, "." * 30, "." * 30,
    "......G.......................",
    "." * 30,
    "...###........................",
    "...###........................",
    "." * 30, "." * 30, "." * 30, "." * 30,
    "....P.........................",
    "##############################",
]


def _unfused_platforms(tiles):
    rects = []
    for r, row in enumerate(tiles):
        for c, ch in enumerate(row):
            if ch == '#':
                rects.append(Platform(8 * c, 8 * r, 8, 8))
    return rects


def test_fusion_reduces_count_and_preserves_trajectory():
    fused = Level.from_dict({'version': 3, 'tiles': TILES})
    rects = _unfused_platforms(TILES)
    unfused = Level.from_dict({
        'version': 1, 'width': 240, 'height': 160,
        'player_spawn': [32, 136], 'gem': [48, 80],
        'platforms': [[p.x, p.y, p.w, p.h] for p in rects],
    })
    # fusion actually merged the 3x2 block and the 30-wide floor
    assert len(fused.platforms) == 2
    assert len(unfused.platforms) == 36
    assert fused.player_spawn == unfused.player_spawn
    assert fused.gem == unfused.gem

    rng = random.Random(2024)
    actions = [rng.randrange(6) for _ in range(10_000)]
    g1, g2 = Game(fused, seed=0), Game(unfused, seed=0)
    for a in actions:
        assert g1.step(a).state == g2.step(a).state
