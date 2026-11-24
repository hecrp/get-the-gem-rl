"""Logical dimensions, hitboxes and limits, in pixels."""
SCREEN_WIDTH = 240
SCREEN_HEIGHT = 160
PLAYER_W = 12
PLAYER_H = 16
GEM_SIZE = 8
EXIT_W = 12
EXIT_H = 16
DEFAULT_MAX_STEPS = 1800
STEPS_PER_SECOND = 60

TILE_SIZE = 8
GRID_CELL = 32
MAX_WORLD_W = 1920
MAX_WORLD_H = 1280

# Sentinel: "no explicit max_steps argument was passed" (see Game / from_level).
FROM_LEVEL = object()