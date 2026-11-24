"""Framebuffer tests: SDL dummy is a guard, not an offscreen requirement."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pygame
import pytest


@dataclass(frozen=True)
class Snapshot:
    player_x: float = 16.0
    player_y: float = 128.0
    gem_x: float = 210.0
    gem_y: float = 120.0
    gem_collected: bool = False
    alive: bool = True
    won: bool = False
    timeout: bool = False
    step_count: int = 0


def rect(x, y, w, h):
    return SimpleNamespace(x=x, y=y, w=w, h=h)


@pytest.fixture
def tiny_game():
    state = Snapshot()
    game = SimpleNamespace(
        state=state,
        level=SimpleNamespace(
            width=240, height=160, player_spawn=(16, 128),
            platforms=(rect(0, 144, 240, 16),),
            hazards=(rect(90, 136, 16, 8),),
            gem=rect(210, 120, 8, 8), exit=rect(225, 128, 12, 16),
        ),
    )
    game.get_state = lambda: game.state
    return game


@pytest.mark.parametrize("headless", [True, False])
def test_rgb_is_offscreen_uint8_y_x_copy(monkeypatch, tiny_game, headless):
    def forbidden(*args, **kwargs):
        pytest.fail("offscreen rendering must not initialize subsystems or a window")
    monkeypatch.setattr(pygame, "init", forbidden)
    monkeypatch.setattr(pygame.display, "init", forbidden)
    monkeypatch.setattr(pygame.display, "set_mode", forbidden)
    from rendering import PygameRenderer
    renderer = PygameRenderer(headless=headless)
    frame = renderer.render_rgb(tiny_game)
    assert frame.shape == (160, 240, 3)
    assert frame.dtype == np.uint8
    assert renderer.draw(tiny_game).get_size() == (240, 160)
    frame[:] = 255
    assert not np.all(renderer.render_rgb(tiny_game) == 255)
    renderer.close()
    renderer.close()


def test_platform_geometry_is_y_x_and_rounded(tiny_game):
    from rendering import PygameRenderer
    tiny_game.level.platforms = (rect(10.6, 140.6, 30, 19),)
    frame = PygameRenderer(headless=True).render_rgb(tiny_game)
    background = frame[0, 0]
    assert np.any(frame[141, 11] != background)
    assert np.any(frame[159, 40] != background)
    np.testing.assert_array_equal(frame[140, 11], background)
    np.testing.assert_array_equal(frame[141, 10], background)
    assert np.any(frame[141, 11] != frame[145, 11])  # top edge


def test_player_exact_hitbox_readonly_deterministic_and_moves(tiny_game):
    from rendering import PygameRenderer
    tiny_game.state = replace(tiny_game.state, player_x=16.6, player_y=50.6)
    renderer = PygameRenderer(headless=True)
    before = tiny_game.state
    frame = renderer.render_rgb(tiny_game)
    background = frame[0, 0]
    assert np.all(np.any(frame[51:67, 17:29] != background, axis=2))
    np.testing.assert_array_equal(frame[50, 17], background)
    np.testing.assert_array_equal(frame[67, 17], background)
    np.testing.assert_array_equal(frame[51, 16], background)
    np.testing.assert_array_equal(frame[51, 29], background)
    assert renderer.render_rgb(tiny_game).tobytes() == frame.tobytes()
    assert tiny_game.state == before
    tiny_game.state = replace(before, player_x=40)
    moved = renderer.render_rgb(tiny_game)
    assert moved.tobytes() != frame.tobytes()
    assert not np.shares_memory(moved, frame)


def test_hazards_draw_spikes_inside_trigger(tiny_game):
    from rendering import PygameRenderer
    frame = PygameRenderer(headless=True).render_rgb(tiny_game)
    background = frame[0, 0]
    assert np.any(frame[136, 94] != background)
    assert np.any(frame[143, 90] != background)
    np.testing.assert_array_equal(frame[136, 90], background)
    np.testing.assert_array_equal(frame[135, 94], background)
    np.testing.assert_array_equal(frame[143, 106], background)


def test_gem_diamond_disappears_when_collected(tiny_game):
    from rendering import PygameRenderer
    renderer = PygameRenderer(headless=True)
    frame = renderer.render_rgb(tiny_game)
    background = frame[0, 0]
    assert np.any(frame[124, 214] != background)
    np.testing.assert_array_equal(frame[120, 210], background)
    tiny_game.state = replace(tiny_game.state, gem_collected=True, won=True)
    collected = renderer.render_rgb(tiny_game)
    assert np.all(collected[120:128, 210:218] == background)


def test_optional_exit_visible_without_mutating_game(tiny_game):
    from rendering import PygameRenderer
    renderer = PygameRenderer(headless=True)
    before = tiny_game.state
    frame = renderer.render_rgb(tiny_game)
    assert np.all(np.any(frame[128:144, 225:237] != frame[0, 0], axis=2))
    tiny_game.level.exit = None
    absent = renderer.render_rgb(tiny_game)
    assert np.all(absent[128:144, 225:237] == absent[0, 0])
    assert tiny_game.state == before


def test_present_is_lazy_nearest_neighbor_and_close_releases_display(tiny_game, monkeypatch):
    from rendering import PygameRenderer
    pygame.display.quit()
    def forbidden(*args, **kwargs):
        pytest.fail("only pygame.display.init is allowed")
    monkeypatch.setattr(pygame, "init", forbidden)
    monkeypatch.setattr(pygame.font, "init", forbidden)
    monkeypatch.setattr(pygame.mixer, "init", forbidden)
    renderer = PygameRenderer(scale=2)
    assert not pygame.display.get_init()
    try:
        logical = renderer.render_rgb(tiny_game)
        renderer.present(tiny_game)
        screen = pygame.display.get_surface()
        assert screen.get_size() == (480, 320)
        actual = pygame.surfarray.array3d(screen).transpose(1, 0, 2)
        expected = logical.repeat(2, axis=0).repeat(2, axis=1)
        np.testing.assert_array_equal(actual, expected)
    finally:
        renderer.close()
        renderer.close()
    assert not pygame.display.get_init()


def test_headless_present_rejected_without_display(tiny_game, monkeypatch):
    from rendering import PygameRenderer
    def forbidden(*args, **kwargs):
        pytest.fail("headless present must not create a window")
    monkeypatch.setattr(pygame.display, "set_mode", forbidden)
    with pytest.raises(RuntimeError, match="headless"):
        PygameRenderer(headless=True).present(tiny_game)


@pytest.mark.parametrize("scale", [0, -1, 1.5, True, "3"])
def test_scale_must_be_positive_integer(scale):
    from rendering import PygameRenderer
    with pytest.raises(ValueError, match="positive integer"):
        PygameRenderer(scale=scale)


def test_grayscale_uses_overflow_safe_integer_luminance(tiny_game):
    from rendering import PygameRenderer
    renderer = PygameRenderer(headless=True)
    rgb = renderer.render_rgb(tiny_game).astype(np.uint32)
    expected = ((299 * rgb[:, :, 0] + 587 * rgb[:, :, 1] + 114 * rgb[:, :, 2]) // 1000).astype(np.uint8)
    gray = renderer.render_grayscale(tiny_game)
    assert gray.shape == (160, 240)
    assert gray.dtype == np.uint8
    np.testing.assert_array_equal(gray, expected)
    gray[:] = 0
    np.testing.assert_array_equal(renderer.render_grayscale(tiny_game), expected)


@pytest.mark.parametrize("flags", [{"alive": False}, {"won": True, "gem_collected": True}])
def test_terminal_state_has_distinct_player_color(tiny_game, flags):
    from rendering import PygameRenderer
    renderer = PygameRenderer(headless=True)
    alive = renderer.render_rgb(tiny_game)
    tiny_game.state = replace(tiny_game.state, **flags)
    terminal = renderer.render_rgb(tiny_game)
    assert not np.array_equal(alive[128:144, 16:28], terminal[128:144, 16:28])


@pytest.mark.parametrize("number", [1, 2, 3])
def test_real_game_render_is_readonly_and_deterministic(number, monkeypatch):
    from pathlib import Path
    from game import Game, Action
    from rendering import PygameRenderer
    path = Path(__file__).resolve().parents[1] / "levels" / f"level_{number:03d}.json"
    game = Game.from_level(path, seed=7)
    twin = Game.from_level(path, seed=7)
    def forbidden(*args, **kwargs):
        pytest.fail("render_rgb must never open a window")
    monkeypatch.setattr(pygame.display, "set_mode", forbidden)
    renderer = PygameRenderer()
    initial = renderer.render_rgb(game)
    for action in [Action.RIGHT_JUMP] * 8 + [Action.LEFT] * 4:
        game.step(action)
        twin.step(action)
        before = game.get_state()
        rgb = renderer.render_rgb(game)
        assert game.get_state() == before == twin.get_state()
        assert rgb.tobytes() == renderer.render_rgb(twin).tobytes()
    assert initial.tobytes() != rgb.tobytes()
    game.reset(seed=7)
    assert initial.tobytes() == renderer.render_rgb(game).tobytes()
