"""Pure, stateless camera and rounding math (standard library only).

``Game.step()`` never consults this module. It lives in ``game/`` (not
``rendering/``) so a future partial symbolic observation can reuse the same
viewport computation without pygame or numpy.
"""
import math

from .constants import PLAYER_H, PLAYER_W
from .entities import Rect


def round_half_up(value: float) -> int:
    """Round towards +infinity on ties; NOT Python's banker's ``round``."""
    return math.floor(value + 0.5)


def _axis(level_size: float, view_size: int, player_pos: float, half_extent: float) -> int:
    if level_size >= view_size:
        centre = player_pos + half_extent
        camera = round_half_up(centre - view_size / 2)
        return max(0, min(camera, int(level_size - view_size)))
    # World narrower than the window: centre it, filling the rest with VOID.
    return -((view_size - int(level_size)) // 2)


def compute_viewport(level_w: float, level_h: float,
                     player_x: float, player_y: float,
                     view_w: int = 240, view_h: int = 160) -> Rect:
    """Top-left (integer) world coordinates of the FOLLOW view window."""
    cam_x = _axis(level_w, view_w, player_x, PLAYER_W / 2)
    cam_y = _axis(level_h, view_h, player_y, PLAYER_H / 2)
    return Rect(cam_x, cam_y, view_w, view_h)