"""View modes: shapes, FOLLOW==crop of FULL, VOID, downscale, cache."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from pathlib import Path

import numpy as np
import pytest

from game import Action, Game
from rendering import PygameRenderer, ViewConfig, ViewMode
from rendering.pygame_renderer import BACKGROUND, VOID

ROOT = Path(__file__).resolve().parents[1]


def level(n):
    return Game.from_level(ROOT / 'levels' / f'level_{n:03}.json', seed=0)


def test_shapes_table_on_levels_1_and_7():
    l1, l7 = level(1), level(7)
    follow = PygameRenderer(headless=True)
    assert follow.render_rgb(l1).shape == (160, 240, 3)
    assert follow.render_rgb(l7).shape == (160, 240, 3)

    fixed = PygameRenderer(headless=True,
                           view=ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320)))
    assert fixed.render_rgb(l1).shape == (320, 480, 3)
    assert fixed.render_rgb(l7).shape == (320, 480, 3)

    world = PygameRenderer(headless=True,
                           view=ViewConfig(mode=ViewMode.FULL, canvas_size=None))
    assert world.render_rgb(l1).shape == (160, 240, 3)   # world 240x160
    assert world.render_rgb(l7).shape == (320, 480, 3)   # world 480x320

    small = PygameRenderer(headless=True,
                           view=ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320),
                                           downscale=2))
    assert small.render_rgb(l7).shape == (160, 240, 3)


@pytest.mark.parametrize('number', [1, 2, 3, 4, 5, 6])
def test_retrocompat_follow_equals_full_canvas_none(number):
    game = level(number)
    follow = PygameRenderer(headless=True)
    full = PygameRenderer(headless=True,
                          view=ViewConfig(mode=ViewMode.FULL, canvas_size=None))
    for a in [Action.RIGHT, Action.RIGHT_JUMP, Action.LEFT, Action.NOOP,
              Action.JUMP, Action.LEFT_JUMP]:
        game.step(a)
        np.testing.assert_array_equal(follow.render_rgb(game), full.render_rgb(game))


def test_follow_is_exact_crop_of_full_on_level_7():
    game = level(7)
    r = PygameRenderer(headless=True, view=ViewConfig(mode=ViewMode.FULL, canvas_size=None))
    r.set_view(ViewConfig(mode=ViewMode.FULL, canvas_size=None))
    for a in [Action.RIGHT] * 40 + [Action.RIGHT_JUMP] * 8 + [Action.LEFT] * 12:
        game.step(a)
        full = r.render_rgb(game)
        r.set_view(ViewConfig())  # FOLLOW
        follow = r.render_rgb(game)
        cam = game.get_viewport(240, 160)
        crop = full[cam.y:cam.y + 160, cam.x:cam.x + 240]
        np.testing.assert_array_equal(follow, crop)
        # put the renderer back to FULL for the next state
        r.set_view(ViewConfig(mode=ViewMode.FULL, canvas_size=None))


def test_full_fixed_canvas_shows_void_outside_small_world():
    game = level(1)  # 240x160
    r = PygameRenderer(headless=True,
                       view=ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320)))
    frame = r.render_rgb(game)
    # inside the world (top-left sky) is BACKGROUND, not VOID
    assert np.array_equal(frame[0, 0], BACKGROUND)
    # outside the world (right of x=240, below y=160) is VOID
    assert np.array_equal(frame[200, 300], VOID)
    assert np.array_equal(frame[0, 300], VOID)
    assert np.array_equal(frame[200, 0], VOID)


def test_full_canvas_smaller_than_world_raises():
    game = level(7)  # 480x320
    r = PygameRenderer(headless=True,
                       view=ViewConfig(mode=ViewMode.FULL, canvas_size=(240, 160)))
    with pytest.raises(ValueError, match='larger'):
        r.render_rgb(game)


def test_downscale_is_block_average_and_rejects_non_divisor():
    game = level(7)
    full = PygameRenderer(headless=True,
                          view=ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320)))
    raw = full.render_rgb(game)  # (320, 480, 3)
    k = 2
    blocks = raw.reshape(320 // k, k, 480 // k, k, 3).astype(np.uint16)
    expected = (blocks.sum(axis=(1, 3)) // (k * k)).astype(np.uint8)

    down = PygameRenderer(headless=True,
                          view=ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320),
                                          downscale=k))
    np.testing.assert_array_equal(down.render_rgb(game), expected)

    bad = PygameRenderer(headless=True,
                         view=ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320),
                                         downscale=3))
    with pytest.raises(ValueError, match='downscale'):
        bad.render_rgb(game)


def test_downscale_matches_naive_loop_implementation():
    # The block average must equal a plain double
    # loop on a small 8x12 frame (k=2 -> 4x6).
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    renderer = PygameRenderer(headless=True,
                              view=ViewConfig(mode=ViewMode.FOLLOW, downscale=2))
    got = renderer._downscaled(frame)
    k = 2
    expected = np.zeros((8 // k, 12 // k, 3), dtype=np.uint8)
    for r in range(8 // k):
        for c in range(12 // k):
            for ch in range(3):
                total = 0
                for dy in range(k):
                    for dx in range(k):
                        total += int(frame[r * k + dy, c * k + dx, ch])
                expected[r, c, ch] = total // (k * k)
    np.testing.assert_array_equal(got, expected)


def test_determinism_and_set_view_do_not_touch_state():
    game = level(7)
    r = PygameRenderer(headless=True)
    before = game.get_state()
    f1 = r.render_rgb(game)
    r.set_view(ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320)))
    f2 = r.render_rgb(game)
    r.set_view(ViewConfig())
    f3 = r.render_rgb(game)
    assert game.get_state() == before
    np.testing.assert_array_equal(f1, f3)
    assert f2.shape == (320, 480, 3)


def test_switching_levels_resizes_and_leaves_no_residue():
    r = PygameRenderer(headless=True,
                       view=ViewConfig(mode=ViewMode.FULL, canvas_size=None))
    g7, g1 = level(7), level(1)
    assert r.render_rgb(g7).shape == (320, 480, 3)
    f1 = r.render_rgb(g1)
    assert f1.shape == (160, 240, 3)
    fresh = PygameRenderer(headless=True,
                           view=ViewConfig(mode=ViewMode.FULL, canvas_size=None))
    np.testing.assert_array_equal(f1, fresh.render_rgb(g1))


def test_renderer_does_not_affect_simulation():
    baseline, follow_game, full_game = level(7), level(7), level(7)
    rf = PygameRenderer(headless=True)
    rfull = PygameRenderer(headless=True,
                           view=ViewConfig(mode=ViewMode.FULL, canvas_size=None))
    actions = [Action.RIGHT, Action.RIGHT_JUMP, Action.LEFT, Action.NOOP,
               Action.JUMP, Action.LEFT_JUMP] * 25
    for a in actions:
        baseline.step(a)
        follow_game.step(a)
        rf.render_rgb(follow_game)
        full_game.step(a)
        rfull.render_rgb(full_game)
        assert follow_game.get_state() == baseline.get_state() == full_game.get_state()
