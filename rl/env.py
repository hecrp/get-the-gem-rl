"""Gymnasium environment over ``game.Game``.

This is the RL layer the engine was designed for: rewards, observations,
frame skipping and level sampling all live here, and ``game/`` and ``pcg/``
are used unchanged. ``terminated = won or dead``; ``truncated = timeout``.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game import Game, Level
from game.constants import FROM_LEVEL, SCREEN_HEIGHT, SCREEN_WIDTH, TILE_SIZE

ROOT = Path(__file__).resolve().parents[1]
BUILTIN_LEVELS = tuple(ROOT / 'levels' / f'level_{i:03d}.json' for i in range(1, 10))

# Grid observation channels.
SOLID, HAZARD, GEM, DOOR, OUTSIDE = range(5)
GRID_CHANNELS = ('solid', 'hazard', 'gem', 'door', 'outside')

# Vector observation layout (all float32).
VEC_FIELDS = (
    'vx',              # velocity_x / move_speed, in [-1, 1]
    'vy',              # velocity_y / max_fall_speed, in [-1.125, 1]
    'grounded',        # 0/1
    'has_gem',         # 0/1
    'needs_door',      # 0/1: win_condition == "exit_with_gem"
    'cell_x',          # player-centre offset inside its tile, [0, 1)
    'cell_y',
    'gem_dx',          # gem centre - player centre, in screens (240 px / 160 px)
    'gem_dy',
    'door_dx',         # door centre - player centre, in screens; 0 without a door
    'door_dy',
    'time_left',       # (max_steps - step_count) / max_steps; 1 without a limit
)
GOAL_FIELDS = slice(7, 11)
MAX_SCREENS = 8.0      # 1920 / 240 = 1280 / 160: the largest world, in screens


def _load_level(source) -> Level:
    if isinstance(source, Level):
        return source
    if isinstance(source, int) and not isinstance(source, bool):
        if not 1 <= source <= len(BUILTIN_LEVELS):
            raise ValueError(f'level: built-in levels are 1..{len(BUILTIN_LEVELS)}')
        return Level.from_json(BUILTIN_LEVELS[source - 1])
    if isinstance(source, (str, Path)):
        return Level.from_json(source)
    raise TypeError(f'level: expected Level, int or path, got {source!r}')


def _level_sampler(levels):
    """Normalise ``levels`` to ``sample(np_random) -> Level``.

    Accepts one level (``Level``, built-in number, path), a sequence of them,
    or any source exposing ``sample(rng)`` such as ``pcg.LevelDistribution``
    and ``pcg.LevelPack``.
    """
    if hasattr(levels, 'sample'):
        return levels.sample
    if isinstance(levels, (list, tuple)):
        if not levels:
            raise ValueError('levels: empty sequence')
        loaded = tuple(_load_level(s) for s in levels)
        if len(loaded) == 1:
            return lambda rng: loaded[0]
        return lambda rng: loaded[int(rng.integers(len(loaded)))]
    level = _load_level(levels)
    return lambda rng: level


def _rasterize(level: Level, pad_x: int, pad_y: int) -> np.ndarray:
    """Static tile channels of ``level`` with a ``pad`` border, shape (C, H, W).

    A tile is marked when a rectangle overlaps it with positive area (strict
    AABB, matching the engine). Outside the world: sides are solid, the top is
    open, the bottom is a fall; all three carry the ``outside`` channel.
    """
    cols, rows = level.width // TILE_SIZE, level.height // TILE_SIZE
    grid = np.zeros((len(GRID_CHANNELS), rows + 2 * pad_y, cols + 2 * pad_x), np.uint8)

    def fill(channel, rects):
        for r in rects:
            c0 = int(math.floor(r.x / TILE_SIZE))
            c1 = int(math.ceil((r.x + r.w) / TILE_SIZE))
            r0 = int(math.floor(r.y / TILE_SIZE))
            r1 = int(math.ceil((r.y + r.h) / TILE_SIZE))
            grid[channel, pad_y + r0:pad_y + r1, pad_x + c0:pad_x + c1] = 1

    fill(SOLID, level.platforms)
    fill(HAZARD, level.hazards)
    if level.exit is not None:
        fill(DOOR, (level.exit,))
    grid[OUTSIDE] = 1
    grid[OUTSIDE, pad_y:pad_y + rows, pad_x:pad_x + cols] = 0
    grid[SOLID, :, :pad_x] = 1
    grid[SOLID, :, pad_x + cols:] = 1
    return grid


def _gem_cells(level: Level):
    g = level.gem
    return (int(g.y // TILE_SIZE), int(math.ceil((g.y + g.h) / TILE_SIZE)),
            int(g.x // TILE_SIZE), int(math.ceil((g.x + g.w) / TILE_SIZE)))


class GemEnv(gym.Env):
    """Get the Gem as a Gymnasium environment.

    Parameters
    ----------
    levels:
        A ``Level``, a built-in level number (1-9), a path, a sequence of
        those, or a level source with ``sample(rng)`` (``pcg.LevelDistribution``,
        ``pcg.LevelPack``). A new level is drawn from ``self.np_random`` on
        every ``reset``; ``reset(options={"level": ...})`` overrides it.
    obs_mode:
        ``"grid"``: ``{"grid": (5, rows, cols) uint8, "vec": (12,) float32}``,
        an egocentric tile crop centred on the player plus kinematic/goal
        features. ``"pixels"``: ``{"pixels": (H, W, 3) uint8, "vec": ...}``
        from the pygame renderer.
    grid_size:
        ``(cols, rows)`` of the grid crop, both odd so the player is centred.
        The default (31, 21) is slightly more than one screen.
    view:
        Pixel view for ``obs_mode="pixels"``: ``"follow"`` (240x160 window on
        the player, partial) or ``"full_small"`` (whole world on a 480x320
        canvas downscaled to 240x160; worlds must fit in 480x320).
    goal_vector:
        If False, the gem/door offsets in ``vec`` are zeroed: the agent must
        find them by looking. The vector keeps its shape.
    frame_skip:
        Engine steps per ``env.step`` (the action is held). Rewards are summed
        and the repeat stops at a terminal step.
    win_reward, gem_reward, death_penalty, step_penalty:
        Reward = ``win_reward`` on victory, ``+ gem_reward`` once when the gem
        is picked up in an ``exit_with_gem`` level (in ``gem`` levels the
        pickup *is* the win), ``- death_penalty`` on death, ``- step_penalty``
        per engine step. Default: sparse, +1 for winning only.
    distance_shaping:
        Coefficient of a potential-based shaping term (Ng, Harada & Russell,
        1999) added to every ``env.step``: ``c * (phi(s') - phi(s))`` with
        ``phi = -(remaining straight-line route length in screens)``: player ->
        gem -> door before the pickup, player -> door after it (player -> gem
        in ``gem`` levels). It telescopes over an episode, so it rewards
        progress without changing which policies are optimal; straight lines
        ignore walls, so it can still mislead around obstacles. 0 disables it.
    max_steps:
        Engine-step limit, as in ``Game`` (default: the level's own, else 1800).
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(self, levels=1, *, obs_mode='grid', grid_size=(31, 21),
                 view='follow', goal_vector=True, frame_skip=4,
                 win_reward=1.0, gem_reward=0.0, death_penalty=0.0,
                 step_penalty=0.0, distance_shaping=0.0, max_steps=FROM_LEVEL,
                 render_mode=None,
                 level_cache_size=64):
        if obs_mode not in ('grid', 'pixels'):
            raise ValueError("obs_mode: expected 'grid' or 'pixels'")
        cols, rows = grid_size
        if cols < 1 or rows < 1 or cols % 2 == 0 or rows % 2 == 0:
            raise ValueError('grid_size: expected two positive odd integers')
        if type(frame_skip) is not int or frame_skip < 1:
            raise ValueError('frame_skip: expected a positive integer')
        if render_mode not in (None, 'rgb_array'):
            raise ValueError("render_mode: expected None or 'rgb_array'")
        if view not in ('follow', 'full_small'):
            raise ValueError("view: expected 'follow' or 'full_small'")

        self._sample_level = _level_sampler(levels)
        self.obs_mode = obs_mode
        self.grid_size = (cols, rows)
        self.view = view
        self.goal_vector = goal_vector
        self.frame_skip = frame_skip
        self.win_reward = float(win_reward)
        self.gem_reward = float(gem_reward)
        self.death_penalty = float(death_penalty)
        self.step_penalty = float(step_penalty)
        self.distance_shaping = float(distance_shaping)
        self.render_mode = render_mode
        self._max_steps = max_steps
        self._pad = (cols // 2, rows // 2)
        self._grids: OrderedDict[int, tuple[Level, np.ndarray]] = OrderedDict()
        self._grid_cache_size = level_cache_size
        self._obs_renderer = None
        self._human_renderer = None
        self.game: Game | None = None

        vec_space = spaces.Box(-MAX_SCREENS, MAX_SCREENS, (len(VEC_FIELDS),), np.float32)
        if obs_mode == 'grid':
            image = spaces.Box(0, 1, (len(GRID_CHANNELS), rows, cols), np.uint8)
            self.observation_space = spaces.Dict({'grid': image, 'vec': vec_space})
        else:
            image = spaces.Box(0, 255, (SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
            self.observation_space = spaces.Dict({'pixels': image, 'vec': vec_space})
        self.action_space = spaces.Discrete(6)

    # -- level bookkeeping -------------------------------------------------

    def _static_grid(self, level: Level) -> np.ndarray:
        key = id(level)
        hit = self._grids.get(key)
        if hit is not None and hit[0] is level:
            self._grids.move_to_end(key)
            return hit[1]
        grid = _rasterize(level, *self._pad)
        self._grids[key] = (level, grid)
        if len(self._grids) > self._grid_cache_size:
            self._grids.popitem(last=False)
        return grid

    @property
    def level(self) -> Level:
        return self.game.level

    # -- gymnasium API -----------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        level = (_load_level(options['level']) if 'level' in options
                 else self._sample_level(self.np_random))
        if self.game is None:
            self.game = Game(level, seed=seed, max_steps=self._max_steps)
        else:
            self.game.reset(seed=seed, level=level)
        self._static = self._static_grid(level)
        self._gem_rows_cols = _gem_cells(level)
        self._had_gem = False
        return self._observation(), self._info(gem_event=False)

    def step(self, action):
        if self.game is None:
            raise RuntimeError('step() called before reset()')
        if not self.action_space.contains(action):
            raise ValueError(f'action: expected an integer in [0, 6), got {action!r}')
        game = self.game
        reward = 0.0
        phi_before = self.potential() if self.distance_shaping else 0.0
        gem_event = False
        engine_steps = 0
        for _ in range(self.frame_skip):
            result = game.step(int(action))
            engine_steps += 1
            reward -= self.step_penalty
            if result.state.gem_collected and not self._had_gem:
                self._had_gem = gem_event = True
                if self.level.win_condition == 'exit_with_gem':
                    reward += self.gem_reward
            if result.done:
                break
        if result.won:
            reward += self.win_reward
        elif result.dead:
            reward -= self.death_penalty
        if self.distance_shaping:
            reward += self.distance_shaping * (self.potential() - phi_before)
        info = self._info(gem_event=gem_event)
        info['engine_steps'] = engine_steps
        return (self._observation(), reward, result.won or result.dead,
                result.timeout, info)

    def potential(self) -> float:
        """Minus the straight-line length, in screens, of the route still to
        run: to the gem (and from it to the door) or, carrying it, to the door."""
        level = self.level
        px, py = self._player_centre()

        def dist(ax, ay, bx, by):
            return math.hypot((bx - ax) / SCREEN_WIDTH, (by - ay) / SCREEN_HEIGHT)

        g = level.gem
        gx, gy = g.x + g.w / 2, g.y + g.h / 2
        e = level.exit
        needs_door = level.win_condition == 'exit_with_gem'
        if needs_door:
            ex, ey = e.x + e.w / 2, e.y + e.h / 2
            if self.game.has_gem:
                return -dist(px, py, ex, ey)
            return -(dist(px, py, gx, gy) + dist(gx, gy, ex, ey))
        return -dist(px, py, gx, gy)

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
        if self._human_renderer is None:
            from rendering import PygameRenderer, ViewConfig, ViewMode
            self._human_renderer = PygameRenderer(
                headless=True, view=ViewConfig(mode=ViewMode.FULL, canvas_size=None))
        return self._human_renderer.render_rgb(self.game)

    def close(self):
        for r in (self._obs_renderer, self._human_renderer):
            if r is not None:
                r.close()
        self._obs_renderer = self._human_renderer = None

    # -- observations ------------------------------------------------------

    def _observation(self):
        vec = self._vector()
        if self.obs_mode == 'grid':
            return {'grid': self._grid(), 'vec': vec}
        return {'pixels': self._pixels(), 'vec': vec}

    def _player_centre(self):
        p = self.game.player
        return p.x + p.w / 2, p.y + p.h / 2

    def _grid(self) -> np.ndarray:
        cols, rows = self.grid_size
        pad_x, pad_y = self._pad
        px, py = self._player_centre()
        # (pr, pc): the world tile under the player centre. The static grid is
        # padded by (pad_y, pad_x), so a crop centred on that tile starts at
        # padded row pr and padded column pc; world tile (wr, wc) lands at crop
        # (wr + pad_y - pr, wc + pad_x - pc).
        pc = int(px // TILE_SIZE)
        pr = int(py // TILE_SIZE)
        static = self._static
        height = static.shape[1]
        if 0 <= pr and pr + rows <= height:
            crop = static[:, pr:pr + rows, pc:pc + cols].copy()
        else:
            # Above the open top or falling below the world: rows beyond the
            # padding are open and outside.
            crop = np.zeros((len(GRID_CHANNELS), rows, cols), np.uint8)
            crop[OUTSIDE] = 1
            world_cols = self.level.width // TILE_SIZE
            left, right = pad_x - pc, pad_x - pc + world_cols
            crop[SOLID, :, :max(left, 0)] = 1         # the side walls extend
            crop[SOLID, :, max(right, 0):] = 1        # at every height
            lo, hi = max(pr, 0), min(pr + rows, height)
            if lo < hi:
                crop[:, lo - pr:hi - pr] = static[:, lo:hi, pc:pc + cols]
        if not self.game.has_gem:
            r0, r1, c0, c1 = self._gem_rows_cols
            r0, r1 = max(r0 + pad_y - pr, 0), min(r1 + pad_y - pr, rows)
            c0, c1 = max(c0 + pad_x - pc, 0), min(c1 + pad_x - pc, cols)
            if r0 < r1 and c0 < c1:
                crop[GEM, r0:r1, c0:c1] = 1
        return crop

    def _pixels(self) -> np.ndarray:
        if self._obs_renderer is None:
            from rendering import PygameRenderer, ViewConfig, ViewMode
            view = (ViewConfig(mode=ViewMode.FOLLOW) if self.view == 'follow' else
                    ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320), downscale=2))
            self._obs_renderer = PygameRenderer(headless=True, view=view)
        return self._obs_renderer.render_rgb(self.game)

    def _vector(self) -> np.ndarray:
        game, level = self.game, self.level
        p, phys = game.player, game.physics
        px, py = self._player_centre()
        g = level.gem
        vec = np.zeros(len(VEC_FIELDS), np.float32)
        vec[0] = p.vx / phys.move_speed
        vec[1] = p.vy / phys.max_fall_speed
        vec[2] = p.grounded
        vec[3] = game.has_gem
        vec[4] = level.win_condition == 'exit_with_gem'
        vec[5] = (px % TILE_SIZE) / TILE_SIZE
        vec[6] = (py % TILE_SIZE) / TILE_SIZE
        if not game.has_gem:
            vec[7] = (g.x + g.w / 2 - px) / SCREEN_WIDTH
            vec[8] = (g.y + g.h / 2 - py) / SCREEN_HEIGHT
        e = level.exit
        if e is not None:
            vec[9] = (e.x + e.w / 2 - px) / SCREEN_WIDTH
            vec[10] = (e.y + e.h / 2 - py) / SCREEN_HEIGHT
        if not self.goal_vector:
            vec[GOAL_FIELDS] = 0.0
        if game.max_steps is not None:
            vec[11] = (game.max_steps - game.step_count) / game.max_steps
        else:
            vec[11] = 1.0
        # Goal offsets are already within +-8 screens; this also bounds the
        # velocity ratios under unusual custom physics.
        return np.clip(vec, -MAX_SCREENS, MAX_SCREENS, out=vec)

    def _info(self, gem_event: bool) -> dict:
        game = self.game
        return {
            'level_name': self.level.name,
            'step_count': game.step_count,
            'has_gem': game.has_gem,
            'gem_collected_event': gem_event,
            'is_success': game.is_won,
        }
