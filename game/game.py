"""Episode lifecycle independent of real time and rendering."""
import random

from .actions import Action, decode_action
from .camera import compute_viewport
from .constants import DEFAULT_MAX_STEPS, FROM_LEVEL
from .collision import overlaps
from .level import Level
from .player import Player
from .state import GameSnapshot, GameState, StepResult
from .physics import PhysicsConfig, integrate


def _probe_grounded(player, level) -> bool:
    """True iff a solid's top face lies exactly under the player's feet and
    overlaps them horizontally. Called only from
    ``reset``; in-step grounding still comes from vertical resolution."""
    foot = player.y + player.h
    for i in level.solids_near(player.x, player.y, player.w, player.h + 1):
        p = level.platforms[i]
        if p.y == foot and player.x < p.x + p.w and player.x + player.w > p.x:
            return True
    return False


class Game:
    def __init__(self, level: Level, seed: int | None = None,
                 max_steps=FROM_LEVEL, physics=None):
        if max_steps is FROM_LEVEL:
            resolved = level.max_steps if level.max_steps is not None else DEFAULT_MAX_STEPS
        else:
            resolved = max_steps
        if resolved is not None and (type(resolved) is not int or resolved < 1):
            raise ValueError('max_steps: expected None or a positive integer')
        self.level = level
        self.max_steps = resolved
        self._from_level_max_steps = max_steps is FROM_LEVEL
        self.physics = PhysicsConfig() if physics is None else physics
        self.rng = random.Random(0 if seed is None else seed)
        self._gem_collected = False
        self._exit_open = False
        self.reset()

    @classmethod
    def from_level(cls, path, seed=None, max_steps=FROM_LEVEL, physics=None):
        return cls(Level.from_json(path), seed=seed, max_steps=max_steps, physics=physics)

    @property
    def is_dead(self) -> bool:
        return not self.player.alive

    @property
    def is_won(self) -> bool:
        return self._won

    @property
    def is_timeout(self) -> bool:
        return self._timeout

    @property
    def is_done(self) -> bool:
        return self.is_dead or self.is_won or self.is_timeout

    @property
    def has_gem(self) -> bool:
        return self._gem_collected

    @property
    def is_exit_open(self) -> bool:
        return self._exit_open

    def reset(self, seed: int | None = None, level: Level | None = None) -> GameState:
        if seed is not None:
            self.rng.seed(seed)
        if level is not None:
            self.level = level
            if self._from_level_max_steps:
                self.max_steps = (level.max_steps if level.max_steps is not None
                                  else DEFAULT_MAX_STEPS)
        self.player = Player(*self.level.player_spawn)
        # Probe the ground once at reset so that the
        # post-reset state matches a player at rest on the first step. step()
        # still derives grounded solely from vertical resolution (C-07).
        self.player.grounded = _probe_grounded(self.player, self.level)
        self.step_count = 0
        self._gem_collected = False
        self._exit_open = False
        self._won = False
        self._timeout = False
        return self.get_state()

    def step(self, action: Action | int) -> StepResult:
        # Terminal no-op deliberately precedes even action validation.
        if not self.is_done:
            direction, jump = decode_action(action)
            integrate(self.player, direction, jump, self.physics, self.level)
            self.step_count += 1
            if (self.player.y > self.level.height or
                    any(overlaps(self.player, h) for h in self.level.hazards)):
                self.player.alive = False
            else:
                self._collect_gem_if_present()
                self._check_win()
                if not self._won and self.max_steps is not None and self.step_count >= self.max_steps:
                    self._timeout = True
        return StepResult(self.get_state(), self.is_done, self.is_won,
                          self.is_dead, self.is_timeout)

    def _collect_gem_if_present(self) -> None:
        """Collect the gem once; it never returns inside an episode."""
        if not self._gem_collected and overlaps(self.player, self.level.gem):
            self._gem_collected = True

    def _check_win(self) -> None:
        """Single victory-rule location. Death is already handled before this."""
        if self.level.win_condition == 'gem':
            self._won = self._gem_collected
        else:
            if (self._gem_collected and self.level.exit is not None
                    and overlaps(self.player, self.level.exit)):
                self._exit_open = True
                self._won = True

    def get_state(self) -> GameState:
        p, g = self.player, self.level.gem
        e = self.level.exit
        return GameState(p.x, p.y, p.vx, p.vy, p.grounded, g.x, g.y,
                         self._gem_collected, p.alive, self._won,
                         self._timeout, self.step_count,
                         None if e is None else e.x,
                         None if e is None else e.y,
                         self._exit_open)

    def save_state(self) -> GameSnapshot:
        p = self.player
        return GameSnapshot(
            level=self.level,
            player_x=p.x, player_y=p.y, velocity_x=p.vx, velocity_y=p.vy,
            grounded=p.grounded, alive=p.alive,
            gem_collected=self._gem_collected, exit_open=self._exit_open,
            won=self._won, timeout=self._timeout, step_count=self.step_count,
            rng_state=self.rng.getstate())

    def load_state(self, snapshot: GameSnapshot) -> None:
        if snapshot.level is not self.level:
            raise ValueError('snapshot: belongs to a different level')
        p = self.player
        p.x, p.y = snapshot.player_x, snapshot.player_y
        p.vx, p.vy = snapshot.velocity_x, snapshot.velocity_y
        p.grounded = snapshot.grounded
        p.alive = snapshot.alive
        self._gem_collected = snapshot.gem_collected
        self._exit_open = snapshot.exit_open
        self._won = snapshot.won
        self._timeout = snapshot.timeout
        self.step_count = snapshot.step_count
        self.rng.setstate(snapshot.rng_state)

    def get_viewport(self, view_w: int = 240, view_h: int = 160):
        """Shortcut to the pure camera viewport for the current player position."""
        p = self.player
        return compute_viewport(self.level.width, self.level.height, p.x, p.y,
                                view_w, view_h)

    def get_state_dict(self) -> dict:
        s = self.get_state()
        e = self.level.exit
        return {
            'player': {'x': s.player_x, 'y': s.player_y,
                       'vx': s.velocity_x, 'vy': s.velocity_y,
                       'grounded': s.grounded, 'alive': s.alive},
            'gem': {'x': s.gem_x, 'y': s.gem_y, 'collected': s.gem_collected},
            'exit': None if e is None else {'x': s.exit_x, 'y': s.exit_y,
                                            'open': s.exit_open},
            'flags': {'won': s.won, 'dead': self.is_dead,
                      'timeout': s.timeout, 'done': self.is_done},
            'win_condition': self.level.win_condition,
            'step_count': s.step_count,
            'world': {'width': self.level.width, 'height': self.level.height},
        }