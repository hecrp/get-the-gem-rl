"""pcg.generator: determinism, golden hashes, structural validity, reversibility,
monotonicity, errors and stdlib isolation."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pcg import PRESETS, GenConfig, generate_level, level_to_dict
from pcg.jump_table import build_jump_table

ROOT = Path(__file__).resolve().parents[1]


def _canonical(level):
    return json.dumps(level_to_dict(level), sort_keys=True, separators=(',', ':'))


def _sha256(level):
    return hashlib.sha256(_canonical(level).encode()).hexdigest()


# ---- determinism -------------------------------------------------------

def test_same_seed_config_gives_identical_levels():
    a = level_to_dict(generate_level(3, GenConfig(cols=90, rows=20, difficulty=1.0)))
    b = level_to_dict(generate_level(3, GenConfig(cols=90, rows=20, difficulty=1.0)))
    assert a == b


def test_determinism_across_processes():
    code = ("import json; from pcg import PRESETS, generate_level, level_to_dict; "
            "from pcg.generator import level_to_dict as l2d; "
            "import hashlib; "
            "lv = generate_level(3, PRESETS['wide']); "
            "print(hashlib.sha256(json.dumps(l2d(lv), sort_keys=True, separators=(',',':')).encode()).hexdigest())")
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == _sha256(generate_level(3, PRESETS['wide']))


# ---- golden hashes -----------------------------------------------------

_GOLDEN = [
    (0, 'tiny', '148f59321adf23f04114c832a7f238c95c915b444ac607bcb9c9faa8841fe60f'),
    (1, 'tiny', '04604d7367a5b9870d5bcad7c158e43789550380a06bc534e5775010ad1dcc09'),
    (2, 'small', '19d46ca57c8c09eefa2b061827fe25075858302de90868e3877cf594cc340154'),
    (3, 'small', '63e6dbcb77cf05801bc801c9765c834f379fb9096dc5e892e303568f19837017'),
    (4, 'wide', '6c2fc004b9a89c52422f0d29552311610be57c098bced7e7ec859efec0562a4d'),
    (5, 'wide', '996ea1a2bb325065eda9048c8737266d1bd3aece66e91f529f1ecc99c722274a'),
    (6, 'tall', 'bfa98d54b03c38425b5775bf1309b963792eff9783585981ef8962f78f34c984'),
    (7, 'tall', '46f220b3172764fd4f0605eed49fd34d9fdf398563bacec0a919052ccd375642'),
    (8, 'large', '8a620e822a9acd7a31f75e314100d321eb9124c9dff6244c57c6078ef26cbbcf'),
    (9, 'huge', '1670ab96103ba5c7e6cd1e269c9ec1cb100f133d2996dd8c5d1fbdaa1e61d5c6'),
]


@pytest.mark.parametrize('seed,preset,digest', _GOLDEN)
def test_golden_hashes(seed, preset, digest):
    assert _sha256(generate_level(seed, PRESETS[preset])) == digest


# ---- structural validity ----------------------------------------------

def _markers(tiles, ch):
    return [(r, c) for r, row in enumerate(tiles) for c, x in enumerate(row) if x == ch]


@pytest.mark.parametrize('preset', ['tiny', 'small', 'wide', 'tall'])
def test_structural_validity(preset):
    cfg = PRESETS[preset]
    for seed in range(50):
        level = generate_level(seed, cfg)
        tiles = level.tiles
        rows = len(tiles)
        assert len(_markers(tiles, 'P')) == 1, f'seed {seed} ({preset}): no single spawn'
        assert len(_markers(tiles, 'G')) == 1, f'seed {seed} ({preset}): no single gem'
        assert len(_markers(tiles, 'D')) == 1, f'seed {seed} ({preset}): no single door'
        # spawn and door sit on the floor (one row above the solid base)
        (pr, pc) = _markers(tiles, 'P')[0]
        (dr, dc) = _markers(tiles, 'D')[0]
        assert pr == rows - 3 and tiles[pr + 1][pc] == '#'
        assert dr == rows - 3 and tiles[dr + 1][dc] == '#'
        # gem is two rows above the last waypoint ledge
        (gr, gc) = _markers(tiles, 'G')[0]
        last = level.metadata['waypoints'][-1]
        assert gr == last[0] - 2


# ---- reversibility -----------------------------------------------------

def _waypoint_gd(a, b):
    return max(b[1] - a[2], a[1] - b[2]) - 1, a[0] - b[0]


def test_critical_path_is_reversible():
    table = build_jump_table().window
    for preset, difficulty in [('tiny', 0.2), ('small', 0.5), ('wide', 0.0), ('tall', 1.0)]:
        cfg = GenConfig(**{**dict(cols=PRESETS[preset].cols, rows=PRESETS[preset].rows),
                           'difficulty': difficulty})
        wmin = round(12 - 9 * difficulty)
        for seed in range(20):
            level = generate_level(seed, cfg)
            wps = level.metadata['waypoints']
            for a, b in zip(wps, wps[1:]):
                g, d = _waypoint_gd(a, b)
                assert (g, d) in table, f'move (g={g},d={d}) out of table'
                assert table[(g, d)] >= wmin, f'forward window too small: {(g, d)}'
                assert (g, -d) in table and table[(g, -d)] >= wmin, f'not reversible: {(g, d)}'


def test_waypoints_correspond_to_solid_ledges():
    # metadata.waypoints describe the real critical path: each ledge is solid
    level = generate_level(5, PRESETS['small'])
    tiles = level.tiles
    for (r, c0, c1) in level.metadata['waypoints']:
        assert all(tiles[r][c] == '#' for c in range(c0, c1 + 1)), f'waypoint [{r},{c0},{c1}] not solid'


# ---- monotonicity ------------------------------------------------------

def test_difficulty_monotonic_windows():
    for preset in ['tiny', 'small']:
        cols, rows = PRESETS[preset].cols, PRESETS[preset].rows
        means = []
        for d in (0.0, 0.5, 1.0):
            cfg = GenConfig(cols=cols, rows=rows, difficulty=d)
            mins = [generate_level(s, cfg).metadata['stats']['min_window'] for s in range(20)]
            means.append(sum(mins) / len(mins))
        assert means[0] > means[1] > means[2], f'{preset}: {means}'


def test_size_monotonic_path_ledges():
    counts = []
    for preset in ['tiny', 'small', 'large']:
        cfg = PRESETS[preset]
        vals = [generate_level(s, cfg).metadata['stats']['path_ledges'] for s in range(10)]
        counts.append(sum(vals) / len(vals))
    assert counts[0] <= counts[1] <= counts[2], f'{counts}'


# ---- errors ------------------------------------------------------------

@pytest.mark.parametrize('cols,rows', [(45, 20), (30, 30), (0, 20), (30, 0)])
def test_non_multiple_sizes_raise(cols, rows):
    with pytest.raises(ValueError):
        GenConfig(cols=cols, rows=rows)


@pytest.mark.parametrize('diff', [-0.1, 1.1, 2.0])
def test_out_of_range_difficulty_raises(diff):
    with pytest.raises(ValueError):
        GenConfig(difficulty=diff)


def test_bad_win_condition_raises():
    with pytest.raises(ValueError):
        GenConfig(win_condition='bogus')


# ---- isolation ---------------------------------------------------------

def test_pcg_does_not_import_pygame_or_numpy():
    code = ("import sys, pcg; "
            "assert 'pygame' not in sys.modules, 'pygame imported'; "
            "assert 'numpy' not in sys.modules, 'numpy imported'")
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_game_and_pcg_clean_subprocess_loads_neither_pygame_nor_numpy():
    # Importing game AND pcg in a
    # clean subprocess must load neither pygame nor numpy.
    code = ("import sys, game, pcg; "
            "assert 'pygame' not in sys.modules, 'pygame imported'; "
            "assert 'numpy' not in sys.modules, 'numpy imported'")
    subprocess.run([sys.executable, '-S', '-c', code], cwd=ROOT, check=True)


def test_pcg_never_uses_the_random_module():
    import pcg
    pkg = Path(pcg.__file__).parent
    src = '\n'.join(f.read_text(encoding='utf-8') for f in pkg.glob('*.py'))
    assert 'import random' not in src
    assert 'from random' not in src