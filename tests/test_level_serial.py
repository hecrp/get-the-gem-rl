"""Level.to_dict / level_hash / canonical_json."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from game import Level
from game.level import canonical_json, level_hash
from pcg import PRESETS, GenConfig, generate_level, level_to_dict

ROOT = Path(__file__).resolve().parents[1]
BUILTIN = sorted((ROOT / 'levels').glob('level_*.json'))

FIELDS = ('platforms', 'hazards', 'gem', 'exit', 'player_spawn', 'win_condition',
          'max_steps', 'width', 'height', 'name', 'version')


def _same(a, b):
    for f in FIELDS:
        assert getattr(a, f) == getattr(b, f), f
    assert a == b
    assert a.metadata == b.metadata and a.extras == b.extras


@pytest.mark.parametrize('path', BUILTIN, ids=lambda p: p.stem)
def test_builtin_round_trip(path):
    level = Level.from_json(path)
    data = level.to_dict()
    json.dumps(data)                         # JSON-native
    again = Level.from_dict(json.loads(canonical_json(data)))
    _same(level, again)
    assert level_hash(again) == level_hash(level)
    assert again.to_dict() == data


def test_generated_round_trip_200():
    configs = [PRESETS['tiny'], PRESETS['small'], PRESETS['wide'], PRESETS['tall'],
               GenConfig(cols=30, rows=20, win_condition='gem')]
    n = 0
    for i in range(200):
        cfg = configs[i % len(configs)]
        level = generate_level(10_000 + i, cfg)
        data = level.to_dict()
        assert data == level_to_dict(level)          # consistent with pcg
        again = Level.from_dict(json.loads(canonical_json(data)))
        _same(level, again)
        assert level_hash(again) == level_hash(level)
        n += 1
    assert n == 200


def test_hash_matches_pcg_golden_digest():
    # level_hash of a generated level is the sha256 of the same canonical text
    # the pcg golden tests use.
    import hashlib
    level = generate_level(0, PRESETS['tiny'])
    text = json.dumps(level_to_dict(level), sort_keys=True, separators=(',', ':'))
    assert level_hash(level) == 'sha256:' + hashlib.sha256(text.encode()).hexdigest()


def test_hash_stable_across_processes():
    code = ("from game import Level; from game.level import level_hash; "
            "print(level_hash(Level.from_json('levels/level_009.json')))")
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == level_hash(Level.from_json(ROOT / 'levels' / 'level_009.json'))
    assert out.startswith('sha256:') and len(out) == 7 + 64


def test_canonical_json_rules():
    assert canonical_json({'b': 1, 'a': [1, 2], 'ñ': 'é'}) == '{"a":[1,2],"b":1,"ñ":"é"}'


def test_tiles_with_explicit_extras_round_trip():
    data = {'version': 3, 'tile_size': 8, 'name': 'mixed',
            'tiles': ['.' * 30] * 18 + ['#' * 30] * 2,
            'player_spawn': [16, 128], 'gem': [200, 130], 'exit': [100, 128],
            'platforms': [[40, 100, 16, 8]], 'hazards': [[120, 140, 8, 4]],
            'custom_field': {'x': 1}}
    level = Level.from_dict(data)
    again = Level.from_dict(level.to_dict())
    _same(level, again)
    assert again.extras == {'custom_field': {'x': 1}}


def test_rect_level_with_fractional_coordinates():
    level = Level.from_dict({'width': 240, 'height': 160, 'player_spawn': [16.5, 128],
                             'gem': [216.25, 132], 'platforms': [[0, 144, 240, 16]]})
    d = level.to_dict()
    assert d['player_spawn'] == [16.5, 128] and d['gem'] == [216.25, 132]
    _same(level, Level.from_dict(d))
