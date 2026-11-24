# Procedural generation (`pcg/`)

The generator produces levels that are **winnable by construction**, from one screen up to 1920×1280 pixels, with a stable seed → level mapping. It uses only the standard library and the engine.

```python
from pcg import PRESETS, GenConfig, LevelDistribution, generate_level

level = generate_level(seed=7, config=GenConfig(cols=90, rows=40, difficulty=0.7))
train = LevelDistribution(PRESETS["small"], start_seed=0, num_levels=500)
test = LevelDistribution(PRESETS["small"], start_seed=1_000_000, num_levels=1000)
```

## How it works

The generator builds the **critical path first** on a grid of one-screen rooms (the Spelunky idea), then adds floor, hazards and distractor platforms:

```text
1 layout (room tree by DFS + anchors) → 2 critical path (reversible ledges)
→ 3 floor + hazards → 4 distractors → 5 emit v3 tiles + metadata
```

- Which jumps are possible is not computed from geometry but **measured by simulating the real engine** (see [`JumpTable`](#jumptable)). The width of each jump's take-off window is used as a difficulty metric.
- The path to the gem and back to the door is always reversible and the floor under it is kept intact (pits only open elsewhere), so no living state is ever unwinnable.
- Each stage draws from its own **substream** (`substream(seed, attempt, "layout" / "path" / …)`), so editing one stage does not shift the others. The whole pipeline uses integer arithmetic.
- An attempt that cannot build a level raises `GenFail` and is retried with `attempt + 1`. After `max_attempts` failures it raises `GenerationError`. With `config.verify`, the guided solver also checks the level and `GenerationError` is raised if it is unsolvable.

### Stable identity

`(GENERATOR_VERSION, seed, config)` always yields the same level on any machine and Python version, thanks to its own SplitMix64 PRNG (one stream per stage, integer arithmetic only). Golden-hash tests pin the output. `GENERATOR_VERSION` is currently `1`; any change that alters the output for some seed must increment it.

### Seed convention

`[0, 10⁶)` for training, `[10⁶, 2·10⁶)` for testing.

## `generate_level`

```python
def generate_level(seed: int, config: GenConfig = GenConfig(),
                   physics: PhysicsConfig | None = None) -> Level:
```

Returns a v3 (tiles) `Level` that is always winnable. See [Levels](levels.md) for the format and the `metadata` the generator writes.

## `GenConfig`

A frozen dataclass.

| Field | Default | Constraint (`ValueError`) |
| --- | --- | --- |
| `cols` | `60` | positive multiple of 30 |
| `rows` | `40` | positive multiple of 20 |
| `difficulty` | `0.5` | `0.0 ≤ d ≤ 1.0` |
| `floor_pits` | `True` | — |
| `ledge_spikes` | `True` | — |
| `distractors` | `True` | — |
| `win_condition` | `"exit_with_gem"` | `"exit_with_gem"` or `"gem"` |
| `max_attempts` | `20` | integer `>= 1` |
| `verify` | `False` | force verification by the solver |

## Presets

```python
PRESETS = {
    'tiny':  GenConfig(cols=30, rows=20, difficulty=0.2),
    'small': GenConfig(cols=60, rows=40),
    'wide':  GenConfig(cols=90, rows=20),
    'tall':  GenConfig(cols=30, rows=60),
    'large': GenConfig(cols=120, rows=80),
    'huge':  GenConfig(cols=240, rows=160, difficulty=0.7),
}
```

| Preset | cols×rows | Size (px) | Difficulty |
| --- | --- | --- | --- |
| `tiny` | 30×20 | 240×160 | 0.2 |
| `small` | 60×40 | 480×320 | 0.5 |
| `wide` | 90×20 | 720×160 | 0.5 |
| `tall` | 30×60 | 240×480 | 0.5 |
| `large` | 120×80 | 960×640 | 0.5 |
| `huge` | 240×160 | 1920×1280 | 0.7 |

## `LevelDistribution` and `LevelPack`

```python
class LevelDistribution:
    def __init__(self, config, start_seed=0, num_levels=0, cache_size=256): ...
    def __len__(self) -> int: ...
    def get(self, index: int): ...        # seed = start_seed + index; LRU cache
    def sample(self, rng): ...            # draws only from the caller's RNG
```

- `num_levels=0` means **unbounded** (63-bit seeds); `num_levels > 0` is the finite range `[start_seed, start_seed + num_levels)`.
- `get(index)` is deterministic and caches by seed with an LRU of `cache_size`. `sample(rng)` consumes **only** the caller's RNG (Procgen style); it knows nothing about episodes or rewards. It accepts a `random.Random` (`randrange` / `getrandbits`) or a NumPy generator (`integers`).
- `start_seed` and `num_levels` must be integers `>= 0`; `get` out of range on a finite set raises `IndexError`.

```python
class LevelPack:
    def __init__(self, path): ...         # path to a JSONL built by build_level_pack
    def __len__(self) -> int: ...
    def get(self, index: int): ...        # Level.from_dict of that line
    def sample(self, rng): ...            # randrange(len) or integers(0, len)
```

`LevelPack` has the same `sample` / `get` / `len` interface and loads lazily on first access.

## `JumpTable`

```python
@dataclass(frozen=True)
class JumpTable:
    window: dict[tuple[int, int], int]      # (g, d) -> number of take-off frames that land
    walkoff: dict[tuple[int, int], bool]    # (g, d) -> walking off the edge lands
def build_jump_table(physics: PhysicsConfig | None = None) -> JumpTable:
```

The generator does not hard-code "gap ≤ 5 tiles, rise ≤ 4". It **measures** real reachability by simulating a micro-world with `game.Game` (a 20-tile source ledge on row 30, a 3-tile target ledge) for each `(g, d)` with `g ∈ [0, 6]` and `d ∈ [-8, 5]`. `window[(g, d)]` is the number of take-off frames (including `-1`, walking off) that land on the target: a direct measure of execution difficulty (Compton & Mateas, 2006).

`build_jump_table` is cached per `PhysicsConfig`; if the physics change it is recomputed. The first call costs about 2 s and is amortised afterwards.

## Metrics

```python
def metrics(level: Level) -> LevelMetrics:
```

`LevelMetrics` (frozen) holds `path_ledges`, `rooms_on_path`, `min_window` and `mean_window` (copied from `metadata.stats`), plus `linearity`, `leniency`, `density` and `backtrack_ratio`, derived from the tiles and waypoints. They are computed in pure Python, without NumPy, so all of `pcg/` stays standard-library only. `tools/pcg_report.py` summarises their distribution with `statistics`.

## Other API

```python
def level_to_dict(level: Level) -> dict:   # canonical JSON with stable key order
class GenFail(Exception): ...              # internal: an attempt produced no valid level
class GenerationError(Exception): ...      # public: max_attempts exhausted or verification failed
```

Minimal example:

```python
from pcg import generate_level, PRESETS, LevelDistribution
from game import Game

level = generate_level(0, PRESETS['small'])          # v3 level, seed 0
game = Game(level)                                   # max_steps resolved from the level
state = game.reset()
result = game.step(2)                                # RIGHT

dist = LevelDistribution(PRESETS['small'], start_seed=0, num_levels=500)
train_level = dist.get(17)                           # seed 17, deterministic
```

## Building packs and reports

```bash
python -m tools.build_level_pack --preset small --start-seed 0 --num 1000 --out levels/packs/small_train.jsonl
python -m tools.pcg_report --preset small --num 1000          # expressive-range report
```

`build_level_pack` checks every level with the solver, and `pcg_report --verify` does too. See [Tools](tools.md).

## Generation speed

Measured on an Apple M4 Pro, Python 3.12, 8 seeds per preset with the jump table already cached:

| Preset | ms / level | Preset | ms / level |
| --- | --- | --- | --- |
| `tiny` | 0.7 | `large` | 44.1 |
| `small` | 2.3 | `huge` | 150.5 |
| `wide` | 1.1 | `tall` | 3.6 |
