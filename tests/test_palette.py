"""rendering.palette: frozen values, stdlib only, used by the pygame renderer (R-08)."""
import subprocess
import sys
from pathlib import Path

from rendering import palette

ROOT = Path(__file__).resolve().parents[1]


def test_palette_values_are_the_previous_constants():
    assert palette.BACKGROUND == (24, 29, 48)
    assert palette.VOID == (0, 0, 0)
    assert palette.PLATFORM == (67, 80, 103)
    assert palette.PLATFORM_TOP == (114, 170, 137)
    assert palette.PLAYER == (255, 209, 102)
    assert palette.PLAYER_DEAD == (180, 70, 91)
    assert palette.PLAYER_WON == (146, 235, 162)
    assert palette.HAZARD == (239, 91, 91)
    assert palette.GEM == (91, 228, 233)
    assert palette.EXIT == (116, 94, 160)
    assert palette.EXIT_EDGE == (48, 39, 67)
    assert palette.EXIT_INTERIOR == (217, 241, 226)
    assert palette.GEM_POLYGON == ((3, 0), (7, 3), (4, 7), (0, 4))
    assert palette.SMALL_GEM_POLYGON == ((1, 0), (3, 1), (2, 3), (0, 2))
    assert palette.CARRIED_GEM_OFFSET == (4, -4) and palette.DOOR_KEY_OFFSET == (4, 6)
    assert palette.PLAYER_SIZE == (12, 16) and palette.PLATFORM_TOP_H == 2
    assert palette.spike_polygon(0, 140, 144, 16) == ((0, 143), (4, 140), (7, 143))
    assert palette.spike_polygon(8, 140, 144, 12) == ((8, 143), (11, 140), (11, 143))


def test_renderer_uses_palette():
    from rendering import pygame_renderer
    for name in ('BACKGROUND', 'VOID', 'PLATFORM', 'PLATFORM_TOP', 'PLAYER', 'PLAYER_DEAD',
                 'PLAYER_WON', 'HAZARD', 'GEM', 'EXIT', 'EXIT_EDGE', 'EXIT_INTERIOR'):
        assert getattr(pygame_renderer, name) is getattr(palette, name)


def test_palette_is_stdlib_only_and_json_native():
    import json
    code = ("import sys, json; import rendering.palette as p; json.dumps(p.as_dict()); "
            "print('pygame' in sys.modules, 'numpy' in sys.modules)")
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
    assert out == ['False', 'False']
    theme = palette.as_dict()
    assert theme['colors']['gem'] == [91, 228, 233]
    assert json.loads(json.dumps(theme)) == theme
