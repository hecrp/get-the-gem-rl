"""Action input contract from the RL layer (numpy scalars, rejected types)."""
import pytest

from game import Action, Game, Level
from game.actions import decode_action


def make_game():
    data = dict(width=240, height=160, player_spawn=[16, 128],
                gem=[216, 132], platforms=[[0, 144, 240, 16]])
    return Game(Level.from_dict(data), max_steps=None)


@pytest.mark.parametrize('scalar', ['int64', 'int32', 'uint8', 'intp'])
def test_step_accepts_numpy_integer_scalars(scalar):
    np = pytest.importorskip("numpy")
    game = make_game()
    before = game.get_state()
    result = game.step(getattr(np, scalar)(Action.RIGHT))
    assert result.state.player_x == before.player_x + 1.5
    assert decode_action(getattr(np, scalar)(3)) == (0, True)


def test_step_rejects_non_integral_without_state_change():
    game = make_game()
    before = game.get_state()
    for bad in (2.0, True, None, '2', [2]):
        with pytest.raises(ValueError):
            game.step(bad)
        assert game.get_state() == before


def test_step_rejects_numpy_array():
    np = pytest.importorskip("numpy")
    game = make_game()
    before = game.get_state()
    with pytest.raises(ValueError):
        game.step(np.array([2]))  # not Integral: the RL layer must index it
    assert game.get_state() == before


def test_decode_action_range_and_bool_still_rejected():
    with pytest.raises(ValueError):
        decode_action(True)
    with pytest.raises(ValueError):
        decode_action(-1)
    with pytest.raises(ValueError):
        decode_action(6)


def test_game_module_stays_numpy_free():
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = ("import sys, game, game.actions; "
            "assert 'numpy' not in sys.modules, 'numpy imported'; "
            "import numbers; "
            "assert issubclass(type(1), numbers.Integral)")
    subprocess.run([sys.executable, '-S', '-c', code], cwd=root, check=True)
