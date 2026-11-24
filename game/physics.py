"""Fixed-step integration, using pixels per step (never wall-clock time)."""
from dataclasses import dataclass, fields
import math
from .constants import PLAYER_W
from .collision import move_horizontal, move_vertical


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    gravity: float = 0.25
    move_speed: float = 1.5
    jump_velocity: float = -4.5
    max_fall_speed: float = 4.0

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{field.name}: expected a finite number')
            if not -PLAYER_W < value < PLAYER_W or not math.isfinite(value):
                raise ValueError(f'{field.name}: magnitude must be finite and < {PLAYER_W}')
            if (value >= 0 if field.name == 'jump_velocity' else value <= 0):
                raise ValueError(f'{field.name}: invalid sign')


def integrate(player, direction: int, jump: bool, config: PhysicsConfig, level):
    player.vx = direction * config.move_speed
    if jump and player.grounded:
        player.vy = config.jump_velocity
        player.grounded = False
    player.vy = min(player.vy + config.gravity, config.max_fall_speed)
    move_horizontal(player, level, level.width)
    move_vertical(player, level)
