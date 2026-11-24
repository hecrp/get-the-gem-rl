"""Detached, hashable observations and step results."""
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class GameState:
    player_x: float
    player_y: float
    velocity_x: float
    velocity_y: float
    grounded: bool
    gem_x: float
    gem_y: float
    gem_collected: bool
    alive: bool
    won: bool
    timeout: bool
    step_count: int
    exit_x: float | None = None
    exit_y: float | None = None
    exit_open: bool = False


@dataclass(frozen=True, slots=True)
class StepResult:
    state: GameState
    done: bool
    won: bool
    dead: bool
    timeout: bool


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    """Opaque full mutable-state capture for ``save_state``/``load_state``."""
    level: object = field(compare=False, repr=False)
    player_x: float
    player_y: float
    velocity_x: float
    velocity_y: float
    grounded: bool
    alive: bool
    gem_collected: bool
    exit_open: bool
    won: bool
    timeout: bool
    step_count: int
    rng_state: object = field(compare=False, repr=False)