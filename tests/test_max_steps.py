"""max_steps precedence and world-size validation."""
import pytest

from game import FROM_LEVEL, Game, Level
from game.constants import DEFAULT_MAX_STEPS, MAX_WORLD_H, MAX_WORLD_W


def level(max_steps=None, **kw):
    data = dict(width=240, height=160, player_spawn=[16, 128],
                gem=[216, 132], platforms=[[0, 144, 240, 16]])
    if max_steps is not None:
        data['max_steps'] = max_steps
    data.update(kw)
    return Level.from_dict(data)


def test_default_uses_level_max_steps():
    assert Game(level(max_steps=100)).max_steps == 100


def test_default_falls_back_to_constant_without_level_limit():
    assert Game(level()).max_steps == DEFAULT_MAX_STEPS == 1800


def test_explicit_argument_overrides_level_limit():
    assert Game(level(max_steps=100), max_steps=500).max_steps == 500


def test_explicit_none_disables_limit_over_level_limit():
    assert Game(level(max_steps=100), max_steps=None).max_steps is None


def test_from_level_uses_same_precedence():
    # level 7 declares max_steps=3600
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    g = Game.from_level(root / 'levels' / 'level_007.json')
    assert g.max_steps == 3600
    assert Game.from_level(root / 'levels' / 'level_007.json',
                           max_steps=None).max_steps is None
    assert Game.from_level(root / 'levels' / 'level_007.json',
                           max_steps=42).max_steps == 42


def test_invalid_max_steps_rejected():
    with pytest.raises(ValueError, match='max_steps'):
        Game(level(), max_steps=0)
    with pytest.raises(ValueError, match='max_steps'):
        Game(level(), max_steps=-5)


def test_level_max_steps_must_be_positive_int():
    with pytest.raises(ValueError, match='max_steps'):
        level(max_steps=0)
    with pytest.raises(ValueError, match='max_steps'):
        level(max_steps='100')


def test_size_must_be_multiple_of_8():
    with pytest.raises(ValueError, match='multiples of 8'):
        level(width=241)


def test_size_minimum_240x160():
    with pytest.raises(ValueError, match='minimum'):
        level(width=232, height=160)


def test_size_maximum_1920x1280():
    assert Game(level(width=MAX_WORLD_W, height=MAX_WORLD_H)).max_steps > 0
    with pytest.raises(ValueError, match='maximum'):
        level(width=MAX_WORLD_W + 8)
    with pytest.raises(ValueError, match='maximum'):
        level(height=MAX_WORLD_H + 8)
