from dataclasses import FrozenInstanceError

import pytest


def base_level(**updates):
    data = dict(width=240, height=160, player_spawn=[16, 128],
                gem=[216, 132], platforms=[[0, 144, 240, 16]])
    data.update(updates)
    return data


def test_level_models_geometry_as_immutable_entities():
    from game.level import Level
    from game.entities import Rect
    level = Level.from_dict(base_level())
    assert level.player_spawn == (16.0, 128.0)
    assert isinstance(level.platforms, tuple)
    assert isinstance(level.gem, Rect)
    assert (level.gem.x, level.gem.y, level.gem.w, level.gem.h) == (216, 132, 8, 8)
    assert level.platforms[0].y == 144
    assert level.hazards == ()
    assert level.exit is None
    with pytest.raises(FrozenInstanceError):
        level.width = 10


@pytest.mark.parametrize("field,value", [
    ("width", 0), ("height", -1), ("width", True), ("height", 160.0),
    ("player_spawn", [0]), ("player_spawn", [float("nan"), 0]),
    ("player_spawn", [-1, 0]), ("player_spawn", [239, 0]),
    ("player_spawn", [16, 140]), ("player_spawn", [216, 128]),
    ("gem", [240, 0]), ("gem", [0, 145]), ("gem", [False, 0]),
    ("platforms", [[0, 144, 0, 16]]), ("platforms", [[0, 144, 240]]),
    ("hazards", [[16, 128, 12, 16]]), ("hazards", None),
    ("exit", [0]), ("version", True), ("name", 4), ("metadata", []),
])
def test_rejects_invalid_level_fields(field, value):
    from game.level import Level
    with pytest.raises(ValueError, match=field):
        Level.from_dict(base_level(**{field: value}))


@pytest.mark.parametrize("field", ["width", "height", "player_spawn", "gem", "platforms"])
def test_required_fields(field):
    from game.level import Level
    data = base_level()
    del data[field]
    with pytest.raises(ValueError, match=field):
        Level.from_dict(data)


def test_json_source_errors_and_optional_data_are_defensive(tmp_path):
    import json
    from game.level import Level
    data = base_level(exit=[200, 128], metadata={"nested": [1]}, doors=[{"x": 9}],
                      hazards=[[100, 140, 5, 4]], name="Test", version=2)
    path = tmp_path / "level.json"
    path.write_text(json.dumps(data))
    level = Level.from_json(path)
    assert level.name == "Test" and level.version == 2
    assert level.exit.w == 12 and level.exit.h == 16
    assert level.hazards[0].x == 100
    level.metadata["nested"].append(2)
    level.extras["doors"][0]["x"] = 0
    data["metadata"]["nested"].append(3)
    assert level.metadata == {"nested": [1]}
    assert level.extras == {"doors": [{"x": 9}]}
    path.write_text('{"width": false}')
    with pytest.raises(ValueError, match=r"level.json.*width"):
        Level.from_json(path)
    path.write_text('{broken')
    with pytest.raises(ValueError, match=r"level.json.*JSON"):
        Level.from_json(path)


def test_direct_construction_validates_and_copies_geometry():
    from game.level import Level
    from game.entities import Gem, Platform
    platforms = [Platform(0, 144, 240, 16)]
    level = Level(240, 160, [16, 128], Gem(216, 132, 8, 8), platforms)
    platforms.clear()
    assert len(level.platforms) == 1
    assert isinstance(level.player_spawn, tuple)
    with pytest.raises(ValueError, match="player_spawn"):
        Level(240, 160, (16, 144), level.gem, level.platforms)


@pytest.mark.parametrize('number,name', [(1, 'Walk'), (2, 'Steps'), (3, 'Pit')])
def test_supplied_levels_have_scripted_solutions(number, name):
    from pathlib import Path
    from game import Game, Action
    path = Path(__file__).resolve().parents[1] / 'levels' / f'level_{number:03}.json'
    game = Game.from_level(path, seed=1)
    assert game.level.name == name
    # The spawn rests on the floor, so reset probes grounded.
    assert game.get_state().grounded
    intervals = {1: (), 2: ((30, 40), (92, 100), (160, 168)),
                 3: ((70, 78), (128, 136), (176, 186))}[number]
    for _ in range(300):
        state = game.get_state()
        jump = state.grounded and any(lo <= state.player_x <= hi for lo, hi in intervals)
        result = game.step(Action.RIGHT_JUMP if jump else Action.RIGHT)
        if result.done:
            break
    assert result.won and not result.dead and not result.timeout


@pytest.mark.parametrize('number,flag', [(2, 'is_timeout'), (3, 'is_dead')])
def test_right_alone_does_not_solve_harder_levels(number, flag):
    from pathlib import Path
    from game import Game, Action
    path = Path(__file__).resolve().parents[1] / 'levels' / f'level_{number:03}.json'
    game = Game.from_level(path)
    while not game.is_done:
        game.step(Action.RIGHT)
    assert getattr(game, flag)


def test_level_validation_cli(tmp_path):
    from pathlib import Path
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[1]
    for number in (1, 2, 3):
        path = root / 'levels' / f'level_{number:03}.json'
        result = subprocess.run([sys.executable, '-m', 'tools.level_loader', str(path)],
                                cwd=root, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert 'OK' in result.stdout and path.name in result.stdout
    bad = tmp_path / 'bad.json'
    bad.write_text('{}')
    result = subprocess.run([sys.executable, '-m', 'tools.level_loader', str(bad)],
                            cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'bad.json' in result.stderr and 'width' in result.stderr
    assert 'Traceback' not in result.stderr
