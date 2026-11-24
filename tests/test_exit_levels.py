"""M7 level format, validation and reference solutions."""
from dataclasses import fields
import json
from pathlib import Path

import pytest

from game import Action, Game, Level
from game.entities import Exit, Gem, Platform

ROOT = Path(__file__).resolve().parents[1]


def level_data(**updates):
    data = dict(width=240, height=160, player_spawn=[60, 128],
                gem=[216, 132], platforms=[[0, 144, 240, 16]])
    data.update(updates)
    return data


@pytest.mark.parametrize('version', [1, 2, 3])
@pytest.mark.parametrize('exit_position,condition,expected', [
    (None, {}, 'gem'), ([16, 128], {}, 'exit_with_gem'),
    ([16, 128], {'win_condition': 'gem'}, 'gem'),
    ([16, 128], {'win_condition': 'exit_with_gem'}, 'exit_with_gem'),
])
def test_resolves_win_condition_for_loader_and_direct_construction(
        version, exit_position, condition, expected):
    data = level_data(version=version, exit=exit_position, **condition)
    loaded = Level.from_dict(data)
    door = None if exit_position is None else Exit(*exit_position, 12, 16)
    direct = Level(240, 160, (60, 128), Gem(216, 132, 8, 8),
                   (Platform(0, 144, 240, 16),), (), door, version,
                   '', {}, {}, **condition)
    assert loaded == direct
    assert loaded.win_condition == expected
    assert 'win_condition' not in loaded.extras
    assert [f.name for f in fields(Level)][-1] == 'win_condition'


@pytest.mark.parametrize('version', [-1, 0, 100, None, True, 1.0, '2'])
def test_rejects_unsupported_format_version(tmp_path, version):
    data = level_data(version=version)
    with pytest.raises(ValueError, match='version'):
        Level.from_dict(data)
    with pytest.raises(ValueError, match='version'):
        Level(240, 160, (60, 128), Gem(216, 132, 8, 8),
              (Platform(0, 144, 240, 16),), version=version)
    path = tmp_path / 'bad_version.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r'bad_version.json.*version'):
        Level.from_json(path)


@pytest.mark.parametrize('condition', ['unknown', '', None, 1, True, [], {}])
def test_invalid_win_conditions_rejected_in_all_construction_paths(tmp_path, condition):
    data = level_data(exit=[16, 128], win_condition=condition)
    with pytest.raises(ValueError, match='win_condition'):
        Level.from_dict(data)
    with pytest.raises(ValueError, match='win_condition'):
        Level(240, 160, (60, 128), Gem(216, 132, 8, 8),
              (Platform(0, 144, 240, 16),), exit=Exit(16, 128, 12, 16),
              win_condition=condition)
    path = tmp_path / 'invalid_condition.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r'invalid_condition.json.*win_condition'):
        Level.from_json(path)


@pytest.mark.parametrize('door', [{}, {'exit': None}])
def test_exit_condition_requires_exit(tmp_path, door):
    data = level_data(win_condition='exit_with_gem', **door)
    with pytest.raises(ValueError, match='exit'):
        Level.from_dict(data)
    with pytest.raises(ValueError, match='exit'):
        Level(240, 160, (60, 128), Gem(216, 132, 8, 8),
              (Platform(0, 144, 240, 16),), win_condition='exit_with_gem')
    path = tmp_path / 'missing_exit.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=r'missing_exit.json.*exit'):
        Level.from_json(path)
