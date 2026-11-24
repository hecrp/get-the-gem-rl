"""Mutable player state, owned exclusively by one Game."""
from dataclasses import dataclass
from .constants import PLAYER_H, PLAYER_W
from .entities import Rect


@dataclass(slots=True)
class Player:
    x: float
    y: float
    w: float = PLAYER_W
    h: float = PLAYER_H
    vx: float = 0.0
    vy: float = 0.0
    grounded: bool = False
    alive: bool = True

    @property
    def rect(self) -> Rect:
        return Rect(self.x, self.y, self.w, self.h)
