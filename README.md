# Get the Gem — a platformer RL environment

**Get the Gem** is a small 2D platform game built to be a reinforcement-learning environment: the agent must reach a gem, carry it back to a door, and avoid spikes and pits on the way. It comes with:

- a **deterministic engine** in pure standard-library Python (no pygame, NumPy or wall clock in the simulation), about 150 000 steps/s on one core;
- a **procedural level generator** whose levels are winnable by construction, from one screen up to 1920×1280 pixels, with a stable seed → level mapping and Procgen-style train/test seed splits;
- a **[Gymnasium](https://gymnasium.farama.org/) environment** (`rl.GemEnv`) with tile-grid or pixel observations;
- a single-file **PPO baseline** and first results, including a small reproduction of the Procgen/CoinRun finding that agents trained on few levels memorise them instead of generalising.

## Quick start

Python **3.11 or later**.

```bash
git clone https://github.com/hecrp/get-the-gem-rl.git
cd get-the-gem-rl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt           # engine, renderer, Gymnasium env
python -m pip install -r requirements-train.txt     # + PyTorch for the PPO baseline
```

With `uv`: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-train.txt`.

`pygame-ce` supplies the `pygame` module; do not install both distributions in the same environment. The engine (`game/`) and the generator (`pcg/`) need no third-party packages; NumPy is used by rendering and the RL layer.

```bash
python play.py --level 7 --view follow --scale 4                     # play it yourself
python -m rl.evaluate --levels 1-9 --per-level --episodes 30         # random-policy baseline
python -m rl.ppo --levels 1-6 --distance-shaping 1.0 --total-steps 1000000   # train PPO
```

## The task

- The player moves left/right and jumps; there is no double jump, coyote time or variable jump height. Horizontal control is immediate, also in the air.
- Touching the **gem** collects it. In levels with a **door** (`win_condition: "exit_with_gem"`), the episode is won by touching the door while carrying the gem; without a door, collecting the gem wins.
- Touching **spikes** or falling below the world kills the player. The sides of the world are solid; the top is open.
- Episodes end on death, victory or a step limit (1 800 steps by default, or the level's own `max_steps`).

There are nine hand-made levels: **Walk**, **Steps** and **Pit** (gem only, one screen); **There and back**, **Up and down** and **Pit, twice** (gem then door, one screen); and **First big one** (480×320), **Long haul** (720×160) and **Sky tower** (240×480). Each has a reference solution in `levels/solutions/`.

## Reinforcement learning (`rl/`)

`rl.GemEnv` is a Gymnasium environment over the engine. Rewards, observations, frame skipping and level sampling live only in `rl/`; `game/` and `pcg/` never import it (or Gymnasium). `terminated = won or dead`, `truncated = timeout`.

```python
from rl import GemEnv
from pcg import PRESETS, LevelDistribution

train = LevelDistribution(PRESETS["small"], start_seed=0, num_levels=200)
env = GemEnv(levels=train, frame_skip=4, gem_reward=0.5)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

- **Levels**: a built-in number (1–9), a path, a `Level`, a list of those, or anything with `sample(rng)` (`LevelDistribution`, `LevelPack`). A level is drawn from `env.np_random` on each `reset`; `reset(options={"level": ...})` forces one (useful for evaluation on held-out seeds).
- **Observation** (`obs_mode="grid"`, the default): a dict with
  - `grid`: `(5, 21, 31)` `uint8`, an egocentric tile crop centred on the player with channels `solid, hazard, gem, door, outside` (side walls count as solid; the gem channel clears once it is carried);
  - `vec`: `(12,)` `float32`: velocity, `grounded`, `has_gem`, `needs_door`, the player's offset inside its tile, gem and door offsets in screens, and the fraction of time left. `goal_vector=False` zeroes the gem/door offsets so the agent must find them by looking.
- `obs_mode="pixels"` replaces `grid` by `pixels`, a `(160, 240, 3)` frame: `view="follow"` (camera on the player, partial) or `view="full_small"` (whole world, for worlds up to 480×320). Both have the same shape, so the same network can compare partial and full observability. Grid mode never imports pygame.
- **Actions**: `Discrete(6)` (`NOOP, LEFT, RIGHT, JUMP, LEFT_JUMP, RIGHT_JUMP`), held for `frame_skip` engine steps (default 4); the repeat stops at a terminal step and `info["engine_steps"]` says how many ran.
- **Reward**: sparse by default, `+win_reward` (1.0) on victory. Optional: `gem_reward` (once, on pickup, only in door levels), `death_penalty`, `step_penalty` (per engine step).
- **Distance shaping** (`distance_shaping=c`): potential-based shaping (Ng, Harada & Russell, 1999), `c·(Φ(s′) − Φ(s))` with Φ = minus the straight-line route still to run (player → gem → door, or player → door once carried), in screens. It telescopes over an episode, so it speeds learning without changing which policies are optimal; straight lines ignore walls, so it can mislead around obstacles.
- **info**: `is_success`, `has_gem`, `gem_collected_event`, `step_count`, `level_name`, `engine_steps`.

### Baselines and speed

```bash
python -m rl.evaluate --levels 1-9 --per-level --episodes 30      # uniform random policy
python -m rl.evaluate --policy right --levels 1-9 --per-level --episodes 1
python -m rl.evaluate --preset small --start-seed 1000000 --num-levels 100
```

Measured on an Apple M4 Pro (one core, `frame_skip=4`): a uniformly random policy wins only level 1, 10 % of the time, and none of the generated `small` levels; holding RIGHT_JUMP wins level 3 and nothing else. Throughput is about 38 000 env steps/s (≈150 000 engine steps/s) in grid mode, 3 400 env steps/s with `pixels`/`follow` and 490 env steps/s with `pixels`/`full_small`.

### PPO baseline (`rl/ppo.py`)

A single-file PPO in the style of CleanRL: a small CNN reads the tile grid, an MLP the feature vector, and a shared actor-critic head sits on top. Timeouts are bootstrapped from the value of the final observation. It uses Apple-silicon `mps` (or CUDA) when available; on the CPU the convolutions are about 15× slower.

```bash
python -m rl.ppo --levels 1-6 --total-steps 3000000 --eval-every 50
python -m rl.ppo --levels 1-6 --total-steps 3000000 --eval-every 50 --distance-shaping 1.0
python -m rl.ppo --preset tiny --num-levels 100 --total-steps 5000000 --distance-shaping 1.0 --eval-every 100
```

Each run writes `runs/<name>/`: `config.json`, `progress.csv` (training curve, plus train/test success at each evaluation), `eval.json` (final success per level, or on training seeds vs held-out seeds from 1 000 000 on) and `model.pt`.

First results (Apple M4 Pro, one seed per setting, 64 evaluation episodes per set; 32 envs × 128 steps, `frame_skip=4`, `gem_reward=0.5`):

| Setting | Steps | Wall time | Result |
| --- | --- | --- | --- |
| Levels 1–6, sparse reward | 3 M | 11 min | 100 % on every level. Evaluation success 2 % at 0.2 M steps, 53 % at 0.6 M, 100 % from 1.0 M. |
| Levels 1–6, `distance_shaping=1.0` | 3 M | 11 min | 100 % on every level, 97 % already at 0.2 M steps (≈5× faster). |
| `tiny`, 10 training levels, shaped | 5 M | 28 min | 100 % on training levels, **31 %** on unseen levels (20 % greedy): memorisation. |
| `tiny`, 100 training levels, shaped | 5 M | 27 min | 100 % train, 98 % unseen (83 % greedy). |
| `tiny`, 1 000 training levels, shaped | 5 M | 29 min | 100 % train, 100 % unseen. |
| `tiny`, unbounded levels, shaped | 5 M | 29 min | 100 % train, 100 % unseen. |

A uniformly random policy wins 2 % of unseen `tiny` levels. The four `tiny` runs ran concurrently. These are single-seed results; `tiny` is one screen at difficulty 0.2, so generalisation saturates by 100 levels, and harder presets, `goal_vector=False` or no shaping should spread the curve out.

## Procedural levels (`pcg/`)

```python
from pcg import PRESETS, GenConfig, LevelDistribution, generate_level

level = generate_level(seed=7, config=GenConfig(cols=90, rows=40, difficulty=0.7))
train = LevelDistribution(PRESETS["small"], start_seed=0, num_levels=500)
test = LevelDistribution(PRESETS["small"], start_seed=1_000_000, num_levels=1000)
```

The generator builds the **critical path first** on a grid of one-screen rooms (the Spelunky idea), then adds floor, hazards and distractor platforms. Which jumps are possible is not computed from geometry but **measured by simulating the real engine** (`pcg/jump_table.py`), and the width of each jump's take-off window is used as a difficulty metric. The path to the gem and back to the door is always reversible and the floor under it is kept intact (pits only open elsewhere), so no living state is ever unwinnable. `tools/build_level_pack.py` checks every level with the solver (`tools/pcg_report.py --verify` does too).

- **Stable identity**: `(GENERATOR_VERSION, seed, config)` always yields the same level on any machine and Python version (own SplitMix64 PRNG, one stream per stage, integer arithmetic only). Golden-hash tests pin the output.
- **Seed convention**: `[0, 10⁶)` for training, `[10⁶, 2·10⁶)` for testing.

| Preset | Size (px) | Difficulty |
| --- | --- | --- |
| `tiny` | 240×160 | 0.2 |
| `small` | 480×320 | 0.5 |
| `wide` | 720×160 | 0.5 |
| `tall` | 240×480 | 0.5 |
| `large` | 960×640 | 0.5 |
| `huge` | 1920×1280 | 0.7 |

```bash
python -m tools.build_level_pack --preset small --start-seed 0 --num 1000 --out levels/packs/small_train.jsonl
python -m tools.pcg_report --preset small --num 1000          # expressive-range report
python -m tools.solve_level levels/level_007.json --out levels/solutions/level_007.json
```

`LevelPack` loads a frozen JSON Lines pack with the same interface as `LevelDistribution`, so a benchmark can be pinned against future generator versions.

## Engine API (`game/`)

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

`Game` accepts a `Level` directly. `Level.from_dict(data)` is useful for code-generated levels; `Level.from_json(path)` reads a file. `Game.from_level(path, seed=None, max_steps=..., physics=None)` is a convenience constructor, and construction already resets the world. `reset(level=...)` switches level in place. `save_state()`/`load_state()` capture and restore the full mutable state (used by the solver).

`get_state()` produces a fresh, immutable, hashable `GameState` snapshot. `get_state_dict()` contains only native JSON-compatible values. `step()` returns a frozen `StepResult` with `state`, `done`, `won`, `dead` and `timeout`, and no reward: rewards belong to the RL layer.

### Engine contract

- Origin: top-left. Positive x points right; positive y points down. Coordinates describe the top-left of hitboxes.
- Units are pixels per step, not real-time seconds. Defaults: horizontal speed 1.5, gravity 0.25, jump velocity −4.5, maximum fall speed 4.0. Hitboxes: player 12×16, gem 8×8, door 12×16.
- Strict AABB overlap: merely touching an edge does not activate a trigger. Platforms are solid. Hazards, the gem and the door are non-solid triggers; the door opens by contact while carrying the gem.
- Death from falling occurs when the player's top edge is below the world (`y > level.height`).
- Terminal precedence: **death > victory > timeout**. `dead == not alive`. At most one terminal reason is true. `won ⇒ gem_collected`; in a door level, `won == exit_open`.
- `max_steps` is `None` or a positive integer, with precedence explicit argument (including `None`) > level > 1800. A timeout occurs exactly at `step_count >= max_steps` unless death or victory occurs in that same step.
- After termination, `step()` is a no-op, including for invalid action values. Active episodes reject invalid actions; NumPy integer scalars are accepted.
- `reset()` returns the exact initial state, without reseeding the local RNG; `reset(seed=n)` also reseeds it. The default seed is fixed (0), not system entropy. At reset, `grounded` is probed, so a resting player is a fixed point of `NOOP` and `JUMP` on the very first step takes off.
- No pygame or NumPy imports, rendering, sleeps, wall-clock calculations or global RNG in `game/`.

## Rendering and views (`rendering/`)

The pygame renderer draws offscreen; no window or audio initialisation is required, rendering never changes the simulation, and the same state always produces identical pixels.

```python
from rendering import PygameRenderer, ViewConfig, ViewMode

renderer = PygameRenderer(headless=True)
follow = renderer.render_rgb(game)                          # (160, 240, 3)
renderer.set_view(ViewConfig(mode=ViewMode.FULL, canvas_size=None))
full = renderer.render_rgb(game)                            # (H, W, 3) = world size
renderer.set_view(ViewConfig(mode=ViewMode.FULL, canvas_size=(480, 320), downscale=2))
small = renderer.render_rgb(game)                           # (160, 240, 3)
```

- **`FOLLOW`** (default) shows a 240×160 window centred on the player, clamped to the world edges, through the stateless camera in `game/camera.py`.
- **`FULL`** draws the whole world: at world size with `canvas_size=None`, or on a fixed canvas (area outside the world in black) so that frame shapes stay constant across levels.
- `downscale=k` applies an exact integer block average. `FOLLOW` is byte-for-byte a crop of `FULL`. Frames are `uint8` with image axes `(height, width, 3)` and do not share memory with the drawing surface.

## Playing

```bash
python play.py
python play.py --level 2
python play.py --level levels/level_003.json --scale 4
python play.py --level 7 --view follow --scale 4
```

| Key | Action |
| --- | --- |
| Left arrow / A | Move left |
| Right arrow / D | Move right |
| Space | Jump (hold to jump again on landing) |
| R | Reset the current level |
| 1 … 9 | Select a built-in level |
| N | Next built-in level, cyclically |
| V | Toggle the view: FULL ⇄ FOLLOW |
| Escape / close window | Quit |

One simulation step is taken per frame at 60 FPS. Winning or dying freezes the episode until reset or level selection. `--view full` (the default) shows the whole world; with no `--scale`, the window is sized to fit within 1280×800.

## Level format

Levels are JSON. Rectangles are `[x, y, width, height]`:

```json
{
  "version": 2,
  "name": "There and back",
  "width": 240,
  "height": 160,
  "player_spawn": [60, 128],
  "gem": [216, 132],
  "exit": [16, 128],
  "win_condition": "exit_with_gem",
  "platforms": [[0, 144, 240, 16]],
  "hazards": []
}
```

Required: `width`, `height`, `player_spawn`, `gem`, `platforms`. Optional: `hazards`, `exit`, `win_condition` (`"gem"` or `"exit_with_gem"`, the default when there is a door), `max_steps`, `name`, `metadata`. Unknown keys are kept in `Level.extras`. Invalid levels raise `ValueError`.

`version: 3` levels may instead use an ASCII **tiles** grid, one string per row (`.` empty, `#` solid, `^` spikes, `P` spawn, `G` gem, `D` door). Adjacent solids fuse into single rectangles, so the physics is unchanged. Sizes are multiples of 8 between 240×160 and 1920×1280.

```json
{
  "version": 3,
  "tile_size": 8,
  "win_condition": "exit_with_gem",
  "max_steps": 3600,
  "tiles": [
    "..............................",
    ".......................G......",
    "....................#####.....",
    ".D..P.......^^................",
    "##############################"
  ]
}
```

Validate a level with `python -m tools.level_loader levels/level_001.json`.

## Project layout

```text
game/        deterministic engine, standard library only
pcg/         procedural level generator (standard library + game)
rendering/   offscreen pygame renderer and views
rl/          Gymnasium environment, baselines and PPO
tools/       level validation, solver, level packs, generator report, benchmark
levels/      nine built-in levels and their reference solutions
play.py      keyboard play
tests/       pytest suite
```

## Tests

```bash
python -m pytest -q                    # full suite (~3 min)
python -m pytest -q -m "not slow"      # fast suite (~25 s)
python -m tools.benchmark --steps 100000
```

The suite covers the engine contract and determinism (including 100 000 seeded random steps across all levels with per-step invariants), exact replay of every reference solution, the generator's golden hashes and solvability, renderer output, the Gymnasium API (`check_env`), the tile grid against an independent rasterisation, and end-to-end PPO smoke runs. Renderer tests use SDL's dummy display.

## Not yet

Multiple seeds and confidence intervals for the PPO results, harder generalisation settings, pixel-based training, learning-curve plots, level editing for curricula (ACCEL-style `mutate`), and packaging as an installable library.
