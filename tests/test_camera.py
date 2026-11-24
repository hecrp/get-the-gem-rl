"""Camera viewport math: pure, standard-library only."""
import subprocess
import sys

from game.camera import compute_viewport, round_half_up
from game.entities import Rect


def test_round_half_up_rounds_ties_away_from_banker_semantics():
    assert round_half_up(2.5) == 3
    assert round_half_up(3.5) == 4
    assert round_half_up(0.5) == 1
    assert round_half_up(-0.5) == 0
    assert round_half_up(-1.5) == -1
    assert round_half_up(2.0) == 2
    assert round_half_up(2.4999) == 2


def test_centered_far_from_edges():
    # Player centre (246, 168) sits at the view centre (120, 80) offset.
    view = compute_viewport(480, 320, 240.0, 160.0, 240, 160)
    assert view == Rect(126, 88, 240, 160)
    assert compute_viewport(480, 320, 354.0, 232.0, 240, 160) == Rect(240, 160, 240, 160)


def test_clamped_at_all_four_edges():
    assert compute_viewport(480, 320, 0.0, 0.0) == Rect(0, 0, 240, 160)
    assert compute_viewport(480, 320, 480.0, 0.0) == Rect(240, 0, 240, 160)
    assert compute_viewport(480, 320, 0.0, 320.0) == Rect(0, 160, 240, 160)
    assert compute_viewport(480, 320, 480.0, 320.0) == Rect(240, 160, 240, 160)


def test_world_smaller_than_window_is_centered_negative():
    assert compute_viewport(200, 120, 100.0, 60.0, 240, 160) == Rect(-20, -20, 240, 160)
    assert compute_viewport(240, 100, 120.0, 50.0, 240, 160) == Rect(0, -30, 240, 160)


def test_single_screen_world_is_always_origin():
    for px, py in ((0.0, 0.0), (120.0, 80.0), (228.0, 144.0)):
        assert compute_viewport(240, 160, px, py) == Rect(0, 0, 240, 160)


def test_above_or_below_world_keeps_camera_inside():
    above = compute_viewport(480, 320, 240.0, -50.0)
    assert above.y == 0
    below = compute_viewport(480, 320, 240.0, 500.0)
    assert below.y == 160


def test_pure_function_same_args_same_result():
    a = compute_viewport(480, 320, 123.5, 77.25)
    b = compute_viewport(480, 320, 123.5, 77.25)
    assert a == b and a is not b


def test_camera_module_imports_no_pygame_nor_numpy():
    code = (
        "import sys\n"
        "import game.camera\n"
        "assert 'pygame' not in sys.modules\n"
        "assert 'numpy' not in sys.modules\n"
    )
    subprocess.run([sys.executable, '-S', '-c', code], check=True)