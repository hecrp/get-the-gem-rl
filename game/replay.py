"""Replays as ``level + actions``: RLE, exact re-simulation and episode events.

The engine is deterministic, so an episode is fully described by its level and
its action sequence. ``simulate`` rebuilds the exact trajectory with the real
``Game`` (standard library only: no Gymnasium, NumPy or pygame), holding each
action for ``frame_skip`` engine steps and stopping at the terminal step, which
is exactly what ``rl.GemEnv.step`` does. Nothing here adds state to
``GameState``/``StepResult``: ``gem_step``, ``win_step``, ``death_xy`` and the
end reason are observed from the outside.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .actions import decode_action
from .collision import overlaps
from .constants import DEFAULT_MAX_STEPS, FROM_LEVEL
from .game import Game
from .level import Level, level_hash
from .physics import PhysicsConfig

# Bits of ``Trajectory.flags`` (one integer per recorded state).
GROUNDED, GEM_COLLECTED, DEAD, WON, EXIT_OPEN = (1, 2, 4, 8, 16)

REPLAY_SCHEMA = 1


def rle_encode(actions: Iterable[int]) -> list[list[int]]:
    """``[3, 3, 5]`` -> ``[[3, 2], [5, 1]]``. Validates every action."""
    out: list[list[int]] = []
    for a in actions:
        decode_action(a)
        a = int(a)
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return out


def rle_decode(pairs: Iterable[Sequence[int]]) -> Iterator[int]:
    """Expand ``[[action, repeat], ...]``; repeats must be integers >= 1 and
    actions are validated with the engine's own rules (``ValueError``)."""
    for i, pair in enumerate(pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f'actions[{i}]: expected [action, repeat]')
        action, repeat = pair
        decode_action(action)
        if isinstance(repeat, bool) or type(repeat) is not int or repeat < 1:
            raise ValueError(f'actions[{i}]: repeat must be an integer >= 1')
        action = int(action)
        for _ in range(repeat):
            yield action


def physics_from_dict(data) -> PhysicsConfig | None:
    """``None``/missing = default physics; else ``PhysicsConfig(**data)``."""
    if data is None:
        return None
    return PhysicsConfig(**data)


def physics_to_dict(physics: PhysicsConfig | None):
    """``None`` for default physics (as stored in replays), else its fields."""
    if physics is None or physics == PhysicsConfig():
        return None
    return {'gravity': physics.gravity, 'move_speed': physics.move_speed,
            'jump_velocity': physics.jump_velocity, 'max_fall_speed': physics.max_fall_speed}


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Result of ``simulate``. ``xy``/``flags`` have one entry per state,
    index 0 being the state right after reset (empty when ``record=False``)."""
    xy: tuple[float, ...]
    flags: tuple[int, ...]
    events: tuple[dict, ...]
    engine_steps: int
    env_steps: int
    result: str                     # 'won' | 'dead' | 'timeout' | 'incomplete'
    end_reason: str | None          # 'won' | 'hazard' | 'fall' | 'timeout' | None
    gem_step: int | None
    win_step: int | None
    death_xy: tuple[float, float] | None
    final_state: object             # game.GameState

    def outcome(self) -> dict:
        """The replay ``outcome`` block (spec 13 §91.3) without ``return``."""
        return {'result': self.result, 'end_reason': self.end_reason,
                'engine_steps': self.engine_steps, 'env_steps': self.env_steps,
                'gem_step': self.gem_step, 'win_step': self.win_step,
                'death_xy': None if self.death_xy is None else list(self.death_xy)}


def _flags(game: Game) -> int:
    p = game.player
    return ((GROUNDED if p.grounded else 0) | (GEM_COLLECTED if game._gem_collected else 0)
            | (0 if p.alive else DEAD) | (WON if game._won else 0)
            | (EXIT_OPEN if game._exit_open else 0))


def first_hazard(level: Level, x: float, y: float, w: float, h: float) -> int | None:
    """Index of the first hazard overlapping the AABB, via the broad phase."""
    from .entities import Rect
    box = Rect(x, y, w, h)
    for i in level.hazards_near(x, y, w, h):
        if overlaps(box, level.hazards[i]):
            return i
    return None


def simulate(level: Level, actions_rle, *, frame_skip: int = 1,
             max_steps=FROM_LEVEL, physics: PhysicsConfig | None = None,
             record: bool = True) -> Trajectory:
    """Replay ``actions_rle`` (RLE over env steps) on ``level``.

    Each env step holds its action for ``frame_skip`` engine steps and stops at
    the terminal step, as ``GemEnv`` does. Actions left over after the episode
    ends are ignored; if the actions run out first the result is
    ``'incomplete'`` (not an error). ``record=False`` skips the per-step
    positions and flags (for bulk sweeps) but still reports every event.
    """
    if type(frame_skip) is not int or frame_skip < 1:
        raise ValueError('frame_skip: expected a positive integer')
    game = Game(level, max_steps=max_steps, physics=physics)
    p = game.player
    xy: list[float] = []
    flags: list[int] = []
    if record:
        xy += (p.x, p.y)
        flags.append(_flags(game))
    events: list[dict] = []
    gem_step = win_step = None
    env_steps = 0
    door_level = level.win_condition == 'exit_with_gem'
    for action in rle_decode(actions_rle):
        if game.is_done:
            break
        env_steps += 1
        for _ in range(frame_skip):
            result = game.step(action)
            if record:
                xy += (p.x, p.y)
                flags.append(_flags(game))
            if gem_step is None and game._gem_collected:
                gem_step = game.step_count
                events.append({'t': gem_step, 'type': 'gem'})
            if result.done:
                break
    reason = game.end_reason
    death_xy = None
    if reason == 'won':
        win_step = game.step_count
        if door_level:
            events.append({'t': win_step, 'type': 'exit_open'})
        events.append({'t': win_step, 'type': 'won'})
    elif reason in ('hazard', 'fall'):
        death_xy = (p.x, p.y)
        ev = {'t': game.step_count, 'type': 'dead', 'cause': reason, 'x': p.x, 'y': p.y}
        if reason == 'hazard':
            ev['hazard'] = first_hazard(level, p.x, p.y, p.w, p.h)
        events.append(ev)
    elif reason == 'timeout':
        events.append({'t': game.step_count, 'type': 'timeout'})
    result = ('incomplete' if reason is None else
              'won' if reason == 'won' else
              'timeout' if reason == 'timeout' else 'dead')
    return Trajectory(xy=tuple(xy), flags=tuple(flags), events=tuple(events),
                      engine_steps=game.step_count, env_steps=env_steps,
                      result=result, end_reason=reason, gem_step=gem_step,
                      win_step=win_step, death_xy=death_xy,
                      final_state=game.get_state())


def resolved_max_steps(level: Level, max_steps=FROM_LEVEL):
    """The engine-step limit a ``Game`` would use (as recorded in replays)."""
    if max_steps is FROM_LEVEL:
        return level.max_steps if level.max_steps is not None else DEFAULT_MAX_STEPS
    return max_steps


def simulate_replay(replay: dict, level: Level, record: bool = True) -> Trajectory:
    """``simulate`` driven by a replay dict (spec 13 §91.3): only ``env`` and
    ``actions`` matter; ``n`` and ``created`` never influence it."""
    env = replay.get('env') or {}
    max_steps = env.get('max_steps', FROM_LEVEL)
    return simulate(level, replay['actions'], frame_skip=env.get('frame_skip', 1),
                    max_steps=max_steps, physics=physics_from_dict(env.get('physics')),
                    record=record)


def reference_replays(solutions_dir, levels_dir=None) -> list[dict]:
    """Virtual ``_reference`` replays, one per ``levels/solutions/level_NNN.json``
    (the RLE format written by ``tools.solve_level``; engine steps, so
    ``frame_skip`` 1). ``levels_dir`` defaults to the solutions' parent."""
    solutions_dir = Path(solutions_dir)
    levels_dir = Path(levels_dir) if levels_dir is not None else solutions_dir.parent
    out = []
    for path in sorted(solutions_dir.glob('level_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        stem = path.stem                          # level_007
        number = int(stem.split('_')[1])
        level = Level.from_json(levels_dir / data.get('level', path.name))
        traj = simulate(level, data['actions'], frame_skip=1, record=False)
        out.append({
            'schema': REPLAY_SCHEMA, 'n': number, 'id': f'_reference/{stem}',
            'run': '_reference', 'created': None,
            'source': {'kind': 'reference', 'file': f'levels/solutions/{path.name}'},
            'policy': 'reference', 'split': 'builtin',
            'level': {'ref': {'kind': 'builtin', 'n': number}, 'hash': level_hash(level),
                      'name': level.name, 'width': level.width, 'height': level.height},
            'env': {'frame_skip': 1, 'max_steps': resolved_max_steps(level), 'physics': None},
            'actions': data['actions'],
            'outcome': traj.outcome(),
            'tags': ['reference'],
        })
    return out
