"""Deterministic, window-independent pixel framebuffer with view modes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pygame

from game.camera import compute_viewport, round_half_up

if TYPE_CHECKING:
    from game import Game

from .palette import (BACKGROUND, CARRIED_GEM_OFFSET, DOOR_EDGE_WIDTH,  # noqa: F401
                      DOOR_KEY_OFFSET, EXIT, EXIT_EDGE, EXIT_INTERIOR, GEM,
                      GEM_POLYGON, HAZARD, PLATFORM, PLATFORM_TOP, PLATFORM_TOP_H,
                      PLAYER, PLAYER_DEAD, PLAYER_SIZE, PLAYER_WON,
                      SMALL_GEM_POLYGON, SPIKE_PERIOD, VOID, spike_polygon)


def _poly(points, ox, oy):
    return [(ox + x, oy + y) for x, y in points]


class ViewMode(Enum):
    FOLLOW = "follow"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class ViewConfig:
    mode: ViewMode = ViewMode.FOLLOW
    viewport_size: tuple = (240, 160)          # (w, h); FOLLOW only
    canvas_size: tuple | None = (480, 320)     # (w, h); FULL only; None = world size
    downscale: int = 1                         # integer factor; render_rgb/grayscale only


def _rect(entity) -> pygame.Rect:
    return pygame.Rect(*(round_half_up(v) for v in
                         (entity.x, entity.y, entity.w, entity.h)))


class PygameRenderer:
    """Draw offscreen by default; the display is reserved for ``present``."""

    def __init__(self, scale: int = 3, headless: bool = False,
                 view: ViewConfig | None = None):
        if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
            raise ValueError("scale must be a positive integer")
        self.scale = scale
        self.headless = headless
        self.view = view if view is not None else ViewConfig()
        self._canvas: pygame.Surface | None = None
        self._static_level = None
        self._static_surface: pygame.Surface | None = None
        self._window: pygame.Surface | None = None

    def set_view(self, view: ViewConfig) -> None:
        self.view = view
        self._canvas = None
        if self._window is not None:
            pygame.display.quit()
            self._window = None

    def _level_size(self, level) -> tuple:
        w = getattr(level, 'width', None)
        h = getattr(level, 'height', None)
        if w is None or h is None:
            # minimal test mocks don't declare a size: assume single screen
            return self.view.viewport_size
        return (w, h)

    def _static_layer(self, level, lw, lh) -> pygame.Surface:
        if self._static_level is not level:
            surf = pygame.Surface((lw, lh))
            surf.fill(BACKGROUND)
            for platform in level.platforms:
                box = _rect(platform)
                pygame.draw.rect(surf, PLATFORM, box)
                pygame.draw.rect(surf, PLATFORM_TOP,
                                 (box.x, box.y, box.w, min(PLATFORM_TOP_H, box.h)))
            for hazard in level.hazards:
                box = _rect(hazard)
                for x in range(box.left, box.right, SPIKE_PERIOD):
                    pygame.draw.polygon(surf, HAZARD,
                                        spike_polygon(x, box.top, box.bottom, box.right))
            self._static_level = level
            self._static_surface = surf
        return self._static_surface

    def draw(self, game: Game) -> pygame.Surface:
        state = game.get_state()
        level = game.level
        lw, lh = self._level_size(level)
        vw, vh = self.view.viewport_size

        if self.view.mode is ViewMode.FULL:
            cam_x = cam_y = 0
            if self.view.canvas_size is None:
                cw, ch = lw, lh
            else:
                cw, ch = self.view.canvas_size
                if cw < lw or ch < lh:
                    raise ValueError(
                        "FULL canvas is smaller than the world; "
                        "use FOLLOW or a larger canvas")
        else:
            cw, ch = vw, vh
            vp = compute_viewport(lw, lh, state.player_x, state.player_y, vw, vh)
            cam_x, cam_y = vp.x, vp.y

        if self._canvas is None or self._canvas.get_size() != (cw, ch):
            self._canvas = pygame.Surface((cw, ch))
        canvas = self._canvas
        canvas.fill(VOID)
        canvas.blit(self._static_layer(level, lw, lh), (-cam_x, -cam_y))

        # Dynamic layer, painted in this order: door → gem → player → carried gem.
        if level.exit is not None:
            door = _rect(level.exit).move(-cam_x, -cam_y)
            condition = getattr(level, 'win_condition', 'gem')
            decorative = condition == 'gem'
            collected = state.gem_collected and not decorative
            opened = getattr(state, 'exit_open', False) and not decorative
            edge = GEM if collected else EXIT_EDGE
            pygame.draw.rect(canvas, EXIT_INTERIOR if opened else EXIT, door)
            pygame.draw.rect(canvas, edge, door, width=DOOR_EDGE_WIDTH)
            if not opened:
                pygame.draw.polygon(canvas, edge,
                                    _poly(SMALL_GEM_POLYGON, door.x + DOOR_KEY_OFFSET[0],
                                          door.y + DOOR_KEY_OFFSET[1]))
        if not state.gem_collected:
            x = round_half_up(state.gem_x) - cam_x
            y = round_half_up(state.gem_y) - cam_y
            pygame.draw.polygon(canvas, GEM, _poly(GEM_POLYGON, x, y))
        px = round_half_up(state.player_x) - cam_x
        py = round_half_up(state.player_y) - cam_y
        player = pygame.Rect(px, py, *PLAYER_SIZE)
        color = PLAYER_DEAD if not state.alive else PLAYER_WON if state.won else PLAYER
        pygame.draw.rect(canvas, color, player)
        if state.gem_collected:
            pygame.draw.polygon(canvas, GEM,
                                _poly(SMALL_GEM_POLYGON, px + CARRIED_GEM_OFFSET[0],
                                      py + CARRIED_GEM_OFFSET[1]))
        return canvas

    def _downscaled(self, frame: np.ndarray) -> np.ndarray:
        k = self.view.downscale
        if k <= 1:
            return frame
        h, w, c = frame.shape
        if h % k or w % k:
            raise ValueError(f"downscale {k} must divide both {w} and {h}")
        # Block average, bit-identical to the reference reshape+sum but split
        # into two reductions over contiguous rows, which numpy executes ~2x
        # faster than one strided sum over axis=(1, 3).
        # uint32 accumulation cannot overflow: 255 * k*k << 2**32.
        blocks = frame.reshape(h // k, k, w, c)
        blocks = blocks.sum(axis=1, dtype=np.uint32)
        blocks = blocks.reshape(h // k, w // k, k, c).sum(axis=2, dtype=np.uint32)
        return (blocks // (k * k)).astype(np.uint8)

    def _frame(self, game: Game) -> np.ndarray:
        """Fresh (H, W, 3) uint8 framebuffer; a single ascontiguousarray of the
        surfarray transpose is the one copy this needs."""
        return np.ascontiguousarray(
            pygame.surfarray.array3d(self.draw(game)).transpose(1, 0, 2))

    def render_rgb(self, game: Game) -> np.ndarray:
        """Independent (height, width, RGB) uint8 snapshot, downscaled if set."""
        return self._downscaled(self._frame(game))

    def render_grayscale(self, game: Game) -> np.ndarray:
        """uint8 integer luminance; widen before multiplying to avoid overflow."""
        rgb = self._downscaled(self._frame(game)).astype(np.uint32)
        luminance = (299 * rgb[:, :, 0] + 587 * rgb[:, :, 1] + 114 * rgb[:, :, 2]) // 1000
        return luminance.astype(np.uint8)

    def present(self, game: Game) -> None:
        """Lazily open and flip a nearest-neighbor window; headless raises."""
        if self.headless:
            raise RuntimeError("present is unavailable for a headless renderer")
        frame = self.draw(game)
        w, h = frame.get_size()
        size = (w * self.scale, h * self.scale)
        if self._window is None or self._window.get_size() != size:
            pygame.display.init()
            self._window = pygame.display.set_mode(size)
            pygame.display.set_caption("Get the Gem")
        pygame.transform.scale(frame, size, self._window)
        pygame.display.flip()

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            self._window = None
