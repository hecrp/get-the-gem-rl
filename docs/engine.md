# Engine (`game/`)

The engine is a headless, deterministic simulation written in pure standard-library Python. It imports no pygame, NumPy, clock or global RNG, and it reaches about 150 000 steps/s on one core (see [Testing and performance](testing.md#performance)).

```python
from game import Action, Game

world = Game.from_level("levels/level_001.json", seed=42)
initial = world.reset()
while not world.is_done:
    result = world.step(Action.RIGHT)

assert result.won
print(world.get_state_dict())
assert world.reset() == initial
```

## Contents

- [Coordinates and units](#coordinates-and-units)
- [Contract summary](#contract-summary)
- [Actions](#actions)
- [`PhysicsConfig`](#physicsconfig)
- [`Game`](#game)
- [Integration and collisions](#integration-and-collisions)
- [State objects](#state-objects)
- [`max_steps` precedence](#max_steps-precedence)
- [`Level`](#level)
- [Determinism](#determinism)
- [What the engine offers an RL layer](#what-the-engine-offers-an-rl-layer)

## Coordinates and units

- Coordinates are in **pixels**, with the origin at the **top-left** of the world. `x` grows to the right and `y` grows downwards.
- `player_x`/`player_y`, `gem_x`/`gem_y` and `exit_x`/`exit_y` are the **top-left corner** of the corresponding hitbox.
- The simulation uses a fixed step (`STEPS_PER_SECOND = 60`). Speeds are in pixels per step, never in time units. The engine never reads the clock.

Constants in `game/constants.py`:

| Constant | Value | Meaning |
| --- | --- | --- |
| `SCREEN_WIDTH` / `SCREEN_HEIGHT` | 240 / 160 | logical screen size (default viewport) |
| `PLAYER_W` / `PLAYER_H` | 12 / 16 | player hitbox |
| `GEM_SIZE` | 8 | gem hitbox side |
| `EXIT_W` / `EXIT_H` | 12 / 16 | door hitbox |
| `DEFAULT_MAX_STEPS` | 1800 | 30 s at 60 steps/s |
| `STEPS_PER_SECOND` | 60 | logical ticks per second |
| `TILE_SIZE` | 8 | tile side |
| `GRID_CELL` | 32 | collision broad-phase cell side |
| `MAX_WORLD_W` / `MAX_WORLD_H` | 1920 / 1280 | maximum world size |

## Contract summary

- Strict AABB overlap: merely touching an edge does not activate a trigger. Platforms are solid. Hazards, the gem and the door are non-solid triggers; the door opens by contact while carrying the gem.
- The sides of the world are solid; the top is open. Death from falling occurs when the player's top edge is below the world (`y > level.height`).
- There is no double jump, coyote time or variable jump height. Horizontal control is immediate, also in the air.
- Terminal precedence: **death > victory > timeout**. `dead == not alive`. At most one terminal reason is true. `won ⇒ gem_collected`; in a door level, `won == exit_open`.
- After termination, `step()` is an idempotent no-op, including for invalid action values, and there is no auto-reset. Active episodes reject invalid actions.
- `reset()` returns the exact initial state without reseeding the local RNG; `reset(seed=n)` also reseeds it. The default seed is fixed (0), not system entropy.
- No pygame or NumPy imports, rendering, sleeps, wall-clock calculations or global RNG in `game/`.

## Actions

`Action` is an `IntEnum` with six stable values. Future actions must be added without renumbering.

| Value | Name | Horizontal (`dx`) | Jump |
| --- | --- | --- | --- |
| 0 | `NOOP` | 0 | no |
| 1 | `LEFT` | −1 | no |
| 2 | `RIGHT` | +1 | no |
| 3 | `JUMP` | 0 | yes |
| 4 | `LEFT_JUMP` | −1 | yes |
| 5 | `RIGHT_JUMP` | +1 | yes |

Decoding lives in one place, `game/actions.py`:

```python
def decode_action(action: Action | int) -> tuple[int, bool]:  # (dx, jump)
```

`decode_action` accepts any `numbers.Integral` that is not a `bool` (so NumPy integer scalars work without importing NumPy) and validates the range through `Action(action)`. It raises `ValueError` for a `bool`, a non-integral value (floats, even `2.0`, strings, `None`, arrays) or an out-of-range integer. The engine has no analogue actions and no `action_space`.

## `PhysicsConfig`

A frozen dataclass in `game/physics.py`:

```python
PhysicsConfig(gravity=0.25, move_speed=1.5, jump_velocity=-4.5, max_fall_speed=4.0)
```

| Field | Default | Validated constraint |
| --- | --- | --- |
| `gravity` | `0.25` | finite number, `0 < v < PLAYER_W` (12) |
| `move_speed` | `1.5` | same |
| `jump_velocity` | `-4.5` | finite, `-12 < v < 0` (must be negative) |
| `max_fall_speed` | `4.0` | `0 < v < 12` |

Construction raises `ValueError` if a value is a `bool`, not an `int`/`float`, not finite, larger than `PLAYER_W` in magnitude, or has the wrong sign. All defaults are multiples of `0.25`, so integration stays exact in binary floating point.

## `Game`

```python
class Game:
    def __init__(self, level: Level, seed: int | None = None,
                 max_steps=FROM_LEVEL, physics=None): ...
    @classmethod
    def from_level(cls, path, seed=None, max_steps=FROM_LEVEL, physics=None): ...
```

- `seed=None` means `random.Random(0)`: a **fixed** seed, not system entropy.
- `physics=None` means `PhysicsConfig()`.
- Construction leaves the game **already reset** (it calls `reset()` internally).
- `from_level` accepts a path; `Game` itself accepts a `Level` directly. `Level.from_dict(data)` is useful for code-generated levels.

Read-only properties:

| Property | Type | Definition |
| --- | --- | --- |
| `is_dead` | `bool` | `not player.alive` |
| `is_won` | `bool` | victory reached |
| `is_timeout` | `bool` | step limit reached |
| `is_done` | `bool` | `is_dead or is_won or is_timeout` |
| `has_gem` | `bool` | the gem has been collected |
| `is_exit_open` | `bool` | the door has been opened |

Public attributes: `level`, `max_steps`, `physics`, `rng` (`random.Random`), `player` and `step_count`.

### `reset`

```python
def reset(self, seed: int | None = None, level: Level | None = None) -> GameState:
```

- A `seed` other than `None` **reseeds** the RNG; without one the RNG is not reseeded (as in Gymnasium).
- A `level` other than `None` **replaces** the level in place. If `max_steps` came from the level, it is recomputed from the new one.
- Returns the initial `GameState`.
- **`grounded` is probed at reset**: it is `True` if and only if a solid's top face is exactly under the player's feet and overlaps the player horizontally. A resting player is therefore a fixed point of `NOOP`, and `JUMP` on the very first step takes off (`vy = -4.25`). Inside `step()`, `grounded` still comes only from vertical collision resolution, and `load_state` restores the saved `grounded` without probing again.

### `step`

```python
def step(self, action: Action | int) -> StepResult:
```

Exact order of operations:

1. **Terminal no-op**, before even validating the action: if `is_done`, return the current state untouched (no `step_count` increment, no flag change).
2. `direction, jump = decode_action(action)`. An invalid action raises `ValueError`, only in an active episode.
3. `integrate(...)`, see [below](#integration-and-collisions).
4. `step_count += 1`.
5. **Death**: if `player.y > level.height` or the player overlaps any hazard, `alive = False` and the remaining phases are skipped (death beats gem and victory in the same step).
6. If alive: collect the gem if overlapping it (once only), then check the win condition. If not won and `max_steps` is set and `step_count >= max_steps`, the episode times out.
7. Return `StepResult(state, done, won, dead, timeout)`.

The win rule has a single home:

```python
if self.level.win_condition == 'gem':
    self._won = self._gem_collected
else:  # exit_with_gem
    if (self._gem_collected and self.level.exit is not None
            and overlaps(self.player, self.level.exit)):
        self._exit_open = True
        self._won = True
```

### Other methods

```python
def get_state(self) -> GameState: ...                  # fresh, immutable snapshot
def get_state_dict(self) -> dict: ...                  # native JSON-compatible values only
def save_state(self) -> GameSnapshot: ...              # complete mutable capture, RNG included
def load_state(self, snapshot: GameSnapshot) -> None: ...
def get_viewport(self, view_w=240, view_h=160): ...    # shortcut to compute_viewport
```

- `save_state`/`load_state` capture the player, `step_count`, the flags and the RNG state (`rng.getstate()`/`setstate()`). `load_state` raises `ValueError` if the snapshot belongs to a different `Level` object (compared by identity). The solver relies on them.
- `get_viewport` returns the `(x, y, w, h)` rectangle of the FOLLOW crop for the current position.
- `get_state_dict()` returns `player`, `gem`, `exit` (`None` without a door), `flags`, `win_condition`, `step_count` and `world`.

## Integration and collisions

```python
def integrate(player, direction, jump, config, level):
    player.vx = direction * config.move_speed
    if jump and player.grounded:                 # jump only from the ground
        player.vy = config.jump_velocity
        player.grounded = False
    player.vy = min(player.vy + config.gravity, config.max_fall_speed)
    move_horizontal(player, level, level.width)
    move_vertical(player, level)
```

Collisions are **strict** AABB (`game/collision.py`): touching an edge is not an overlap. Resolution queries the level's broad-phase and is **bit-identical** to brute force over `platforms`.

- `move_horizontal` applies `vx`, resolves lateral overlaps (setting `vx = 0`) and clamps to `[0, width - w]`: the world's sides are solid, the top is open.
- `move_vertical` sets `grounded = False`, applies `vy`, and depending on the sign of the displacement either lands (`grounded = True`, `vy = 0`) or hits a ceiling (`vy = 0`).

## State objects

### `GameState`

A frozen dataclass in `game/state.py`, hashable and compared with `==`:

```python
class GameState:
    player_x: float;  player_y: float
    velocity_x: float;  velocity_y: float
    grounded: bool
    gem_x: float;  gem_y: float;  gem_collected: bool
    alive: bool;  won: bool;  timeout: bool
    step_count: int
    exit_x: float | None = None
    exit_y: float | None = None
    exit_open: bool = False
```

`get_state()` returns a new snapshot; mutating the game afterwards does not alter earlier ones. In `win_condition == "gem"` levels, `exit_x`/`exit_y` are `None` and `exit_open` is always `False`.

### `StepResult`

```python
@dataclass(frozen=True, slots=True)
class StepResult:
    state: GameState
    done: bool
    won: bool
    dead: bool
    timeout: bool
```

Invariants checked by the test suite:

- `done == (won or dead or timeout)`.
- At most one of `{won, dead, timeout}` is `True` (priority `dead > won > timeout`).
- `dead == (not state.alive)`; `won ⇒ state.gem_collected`.
- `step()` after `done` returns the same terminal result.

There is no reward field: rewards belong to the RL layer.

### `GameSnapshot`

An opaque capture for `save_state`/`load_state`: the level, player position and velocity, `grounded`, `alive`, `gem_collected`, `exit_open`, `won`, `timeout`, `step_count` and `rng_state`. It is **not** JSON-serialisable, because it holds the `Level` object and the RNG tuple. To persist a game, store `get_state_dict()` plus the seed, or pickle it.

## `max_steps` precedence

- `Game(..., max_steps=FROM_LEVEL)` (the default) uses `level.max_steps`, falling back to `DEFAULT_MAX_STEPS` (1800) when the level defines none. `FROM_LEVEL` is a sentinel defined in `constants.py`.
- `Game(..., max_steps=None)` means **no limit** (used by `play.py` and the solver).
- `Game(..., max_steps=N)` is an explicit limit.
- Validation: `None` or an integer `>= 1`; anything else raises `ValueError`.

A timeout occurs exactly when `step_count >= max_steps`, unless death or victory occurs in that same step.

## `Level`

A frozen dataclass in `game/level.py`. Geometry is **immutable** and tuple-based; `metadata` and `extras` are stored as a private deep copy and returned as defensive copies, so outside code cannot mutate a shared level.

| Field | Type | Notes |
| --- | --- | --- |
| `width`, `height` | `int` | multiples of 8, between 240×160 and 1920×1280 |
| `player_spawn` | `tuple[float, float]` | top-left corner of the hitbox |
| `gem` | `Gem` | 8×8 |
| `platforms` | `tuple[Platform, ...]` | solid rectangles |
| `hazards` | `tuple[Hazard, ...]` | spikes (default `()`) |
| `exit` | `Exit \| None` | 12×16; `None` without a door |
| `version` | `int` | `1`, `2` or `3` (default `1`) |
| `name` | `str` | default `""` |
| `max_steps` | `int \| None` | default `None` |
| `tiles` | `tuple[str, ...] \| None` | set when loaded from tiles (v3) |
| `win_condition` | `str` | `"gem"` or `"exit_with_gem"` |

If `win_condition` is omitted it is resolved: `"gem"` when there is no door, `"exit_with_gem"` otherwise. Invalid levels raise `ValueError`: bad version, size not a multiple of 8 or out of range, unknown `win_condition`, `exit_with_gem` without a door, wrong gem/door hitbox, a door out of bounds or overlapping platforms/gem/hazards, a spawn out of bounds or overlapping platforms/hazards/gem, a gem fully inside a platform, or wrongly typed fields.

The broad-phase is a uniform grid of `GRID_CELL = 32` cells built at construction. `solids_near(x, y, w, h)` and `hazards_near(x, y, w, h)` return ascending, deduplicated indices of a superset of anything that can overlap the queried rectangle.

The file format is described in [Levels](levels.md).

## Serialisation, end reasons and replays

Additive helpers; none of them changes `step()` or the state objects.

- `Level.to_dict()` returns JSON-native data such that `Level.from_dict(level.to_dict())` is an equal level (tile levels keep their tiles, as `pcg.level_to_dict` does; others use rectangles). `game.level.canonical_json(obj)` is `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` and `game.level.level_hash(level)` is `"sha256:" + hexdigest` of the canonical text: a content address that is stable across processes and Python versions.
- `Game.end_reason` is `None` while the episode runs, else `'hazard'`, `'fall'` (dead with the player below the world), `'won'` or `'timeout'`, derived from existing state with the precedence dead > won > timeout.
- `game.replay.simulate(level, actions_rle, *, frame_skip=1, max_steps=FROM_LEVEL, physics=None, record=True)` replays RLE actions (`[[action, repeat], ...]`, each env step holding its action for `frame_skip` engine steps and stopping at the terminal step, exactly as `GemEnv`) and returns a `Trajectory`: positions and flags per state (index 0 = after reset), events (`gem`, `exit_open`, `won`, `dead` with its cause and hazard index, `timeout`), `result` (`'won'`, `'dead'`, `'timeout'` or `'incomplete'` if the actions run out), `end_reason`, `gem_step`, `win_step`, `death_xy` and the final `GameState`. `rle_encode`/`rle_decode` validate actions with the engine's rules. `reference_replays(solutions_dir)` turns `levels/solutions/` into replay records.
- `game.replay_store` holds the file conventions shared by recorders and readers (atomic JSON, tolerant JSON Lines, the level store and locked replay numbering). See [Dashboard](dashboard.md#data-formats).

## Determinism

**Guaranteed**

- `Game(seed=0)` plus the same action sequence gives exactly the same state sequence. The default physics values are dyadic, and integration is a fixed chain of operations.
- `seed=None` means a fixed seed (0), so determinism is the default.
- `generate_level(seed, config)` uses its own `SplitMix64`, never the `random` module, so it is identical across Python versions and machines.
- `render_rgb` and `render_grayscale` are deterministic: constant colours, integer or exactly-defined NumPy arithmetic.

**Not guaranteed**

- Custom physics constants that are not dyadic, or coordinates outside what the generator produces, may not be exact in real arithmetic. Replays within the same runtime still follow the same operation order.
- The window path `present()` scales with nearest-neighbour `pygame.transform.scale`. It is deterministic in SDL but is not used for training observations.

**How it is tested**: two `Game` instances with the same seed run the same actions and their `GameState`s are compared with `==`; the generator is compared against SHA-256 golden vectors of its canonical JSON; and 100 000 seeded random steps run across all levels with per-step invariants.

## What the engine offers an RL layer

The engine has no `reward`, `action_space` or `observation_space`. This table maps Gymnasium concepts to what it does provide; [`GemEnv`](reinforcement-learning.md) builds on exactly these.

| Gymnasium / training concept | Engine source |
| --- | --- |
| discrete action space | `Action` (6 values) + `decode_action` |
| `terminated` (death / victory) | `StepResult.dead` / `StepResult.won` |
| `truncated` (timeout) | `StepResult.timeout` |
| symbolic observation / `info` | `game.get_state()` or `get_state_dict()` |
| `reward` | none: computed in the RL layer |
| reseed on reset | `game.reset(seed=n)` |
| level change per episode | `game.reset(level=level)`, or `LevelDistribution.get(i)` / `.sample(rng)` |
| "gem collected" event | `gem_collected` goes `False → True` between consecutive states |
| "door opened" event | `exit_open` goes `False → True` |
| visual observation | `PygameRenderer.render_rgb(game)`, see [Rendering](rendering.md) |

Events are derived by comparing two consecutive states; the engine does not emit them as signals.
