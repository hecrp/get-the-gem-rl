"""Immutable static geometry; coordinates are top-left corners."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


class Platform(Rect):
    __slots__ = ()


class Hazard(Rect):
    __slots__ = ()


class Gem(Rect):
    __slots__ = ()


class Exit(Rect):
    __slots__ = ()
