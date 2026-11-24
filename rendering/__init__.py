"""Optional pygame presentation; never imported by the simulation.

``rendering.palette`` is standard library only; the pygame renderer is loaded
lazily on first use of ``PygameRenderer``/``ViewConfig``/``ViewMode``, so that
importing the palette (the web dashboard does) never imports pygame.
"""
__all__ = ["PygameRenderer", "ViewConfig", "ViewMode"]


def __getattr__(name):
    if name in __all__:
        from . import pygame_renderer
        return getattr(pygame_renderer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
