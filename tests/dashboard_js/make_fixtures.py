"""Render the renderer's polygons with pygame and store their pixels, so the
dashboard's JS scanline port can be checked against pygame (node --test).

Only the shape family the renderer draws is covered (the gem, the small gem
used for the carried gem and the door key, and spike triangles of every
height up to 48 px and every partial width): the port reproduces pygame-ce
exactly on these, not on arbitrary polygons.

    python tests/dashboard_js/make_fixtures.py
"""
import json
import os
import random
from pathlib import Path

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import pygame  # noqa: E402

from rendering.palette import GEM_POLYGON, SMALL_GEM_POLYGON, spike_polygon  # noqa: E402


def pixels(points):
    surf = pygame.Surface((128, 128))
    surf.fill((0, 0, 0))
    pygame.draw.polygon(surf, (255, 255, 255), [(x + 32, y + 32) for x, y in points])
    return sorted([x - 32, y - 32] for x in range(128) for y in range(128) if surf.get_at((x, y))[0])


def main():
    cases = [list(GEM_POLYGON), list(SMALL_GEM_POLYGON)]
    cases += [list(spike_polygon(0, 0, h, w)) for h in range(1, 49) for w in range(1, 9)]
    # Translation invariance (the renderer draws these at every position).
    rng = random.Random(7)
    for _ in range(20):
        ox, oy = rng.randrange(-20, 20), rng.randrange(-20, 20)
        cases.append([(x + ox, y + oy) for x, y in GEM_POLYGON])
    out = [{'points': [list(p) for p in c], 'pixels': pixels(c)} for c in cases]
    path = Path(__file__).parent / 'fixtures' / 'polygons.json'
    path.write_text(json.dumps(out, separators=(',', ':')))
    print(f'{path}: {len(out)} polygons')


if __name__ == '__main__':
    main()
