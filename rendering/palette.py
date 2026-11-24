"""Colours and shape rules of the renderer, as plain data (standard library only).

``pygame_renderer`` draws with these values, and the web dashboard serves
``as_dict()`` (``GET /api/theme``) so its canvas renderer uses the very same
palette and pixel rules: positions are rounded with ``floor(v + 0.5)``
(``game.camera.round_half_up``) and polygons are filled with pygame's scanline
rule. Importing this module never imports pygame.
"""
from __future__ import annotations

BACKGROUND = (24, 29, 48)
VOID = (0, 0, 0)
PLATFORM = (67, 80, 103)
PLATFORM_TOP = (114, 170, 137)
PLAYER = (255, 209, 102)
PLAYER_DEAD = (180, 70, 91)
PLAYER_WON = (146, 235, 162)
HAZARD = (239, 91, 91)
GEM = (91, 228, 233)
EXIT = (116, 94, 160)
EXIT_EDGE = (48, 39, 67)
EXIT_INTERIOR = (217, 241, 226)

COLORS = {
    'background': BACKGROUND, 'void': VOID, 'platform': PLATFORM,
    'platform_top': PLATFORM_TOP, 'player': PLAYER, 'player_dead': PLAYER_DEAD,
    'player_won': PLAYER_WON, 'hazard': HAZARD, 'gem': GEM, 'exit': EXIT,
    'exit_edge': EXIT_EDGE, 'exit_interior': EXIT_INTERIOR,
}

# -- shape rules (pixels; offsets from the rounded top-left of the entity) -----

PLAYER_SIZE = (12, 16)
PLATFORM_TOP_H = 2                          # min(2, h) rows of PLATFORM_TOP
GEM_POLYGON = ((3, 0), (7, 3), (4, 7), (0, 4))
CARRIED_GEM_OFFSET = (4, -4)                # from the player's top-left
SMALL_GEM_POLYGON = ((1, 0), (3, 1), (2, 3), (0, 2))   # carried gem and door key
DOOR_KEY_OFFSET = (4, 6)                    # from the door's top-left; drawn while closed
DOOR_EDGE_WIDTH = 1
SPIKE_PERIOD = 8                            # one triangle every 8 px of hazard width
SPIKE_SPAN = 7                              # right corner at x + 7 (clipped to the box)
SPIKE_TIP = 4                               # tip at x + 4 (clipped to the right corner)
# Dynamic layer order: door -> gem -> player -> carried gem.
DRAW_ORDER = ('static', 'door', 'gem', 'player', 'carried_gem')


def spike_polygon(x: int, top: int, bottom: int, right_edge: int):
    """The triangle for the spike segment starting at ``x`` of a hazard box
    ``[left, right_edge)`` x ``[top, bottom)``, exactly as the renderer draws it."""
    right = min(x + SPIKE_SPAN, right_edge - 1)
    tip = min(x + SPIKE_TIP, right)
    return ((x, bottom - 1), (tip, top), (right, bottom - 1))


def as_dict() -> dict:
    """JSON-native palette and shape rules (served at ``GET /api/theme``)."""
    return {
        'colors': {k: list(v) for k, v in COLORS.items()},
        'shapes': {
            'player_size': list(PLAYER_SIZE),
            'platform_top_h': PLATFORM_TOP_H,
            'gem_polygon': [list(p) for p in GEM_POLYGON],
            'small_gem_polygon': [list(p) for p in SMALL_GEM_POLYGON],
            'carried_gem_offset': list(CARRIED_GEM_OFFSET),
            'door_key_offset': list(DOOR_KEY_OFFSET),
            'door_edge_width': DOOR_EDGE_WIDTH,
            'spike_period': SPIKE_PERIOD, 'spike_span': SPIKE_SPAN, 'spike_tip': SPIKE_TIP,
            'draw_order': list(DRAW_ORDER),
        },
        'rules': {'round': 'floor(v + 0.5)', 'polygon_fill': 'pygame scanline'},
    }
