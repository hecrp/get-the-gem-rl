"""Exit appearance and carried-gem framebuffer contracts."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

from game.entities import Rect
from rendering import PygameRenderer
from rendering.pygame_renderer import BACKGROUND, GEM


@dataclass(frozen=True)
class Snapshot:
    player_x: float = 120.0
    player_y: float = 100.0
    gem_x: float = 200.0
    gem_y: float = 100.0
    gem_collected: bool = False
    exit_open: bool = False
    alive: bool = True
    won: bool = False


@pytest.fixture
def scene():
    game = SimpleNamespace(
        state=Snapshot(),
        level=SimpleNamespace(exit=Rect(20, 80, 12, 16),
                              win_condition="exit_with_gem", platforms=(), hazards=()),
    )
    game.get_state = lambda: game.state
    return game


def test_three_door_appearances_differ_inside_exact_door_crop(scene):
    renderer = PygameRenderer(headless=True)
    crops = []
    for collected, opened in [(False, False), (True, False), (True, True)]:
        scene.state = replace(scene.state, gem_collected=collected, exit_open=opened)
        frame = renderer.render_rgb(scene)
        crops.append(frame[80:96, 20:32])
        assert frame.shape == (160, 240, 3)
        assert frame.dtype == np.uint8
        assert np.all(frame[79, 19:33] == BACKGROUND)
        assert np.all(frame[96, 19:33] == BACKGROUND)
        assert np.all(frame[80:96, 19] == BACKGROUND)
        assert np.all(frame[80:96, 32] == BACKGROUND)
    closed, unlockable, opened = crops
    assert len({crop.tobytes() for crop in crops}) == 3
    assert np.all(np.any(closed != BACKGROUND, axis=2))  # opaque leaf
    assert closed[0, 0].sum() < closed[2, 2].sum()  # dark frame
    np.testing.assert_array_equal(unlockable[0, 0], GEM)
    assert np.any(np.all(unlockable[2:-2, 2:-2] == GEM, axis=2))  # keyhole
    assert opened[4, 4].sum() > closed[4, 4].sum()


@pytest.mark.parametrize("legacy", [False, True])
def test_decorative_door_stays_closed_even_with_collected_or_open_flags(scene, legacy):
    if legacy:
        del scene.level.win_condition
    else:
        scene.level.win_condition = "gem"
    renderer = PygameRenderer(headless=True)
    closed = renderer.render_rgb(scene)[80:96, 20:32]
    for opened in (False, True):
        scene.state = replace(scene.state, gem_collected=True, exit_open=opened)
        np.testing.assert_array_equal(renderer.render_rgb(scene)[80:96, 20:32], closed)
