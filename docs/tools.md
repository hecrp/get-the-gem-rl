# Tools

Command-line tools are invoked with `python -m tools.<name>` (or `tools/<name>.py`). The interactive player is `python play.py`. Use `--help` on any of them for the full option list.

| Tool | Purpose |
| --- | --- |
| [`play.py`](#playpy) | play a level with the keyboard |
| [`tools.level_loader`](#toolslevel_loader) | validate a level JSON file |
| [`tools.solve_level`](#toolssolve_level) | solve a level and write a reference solution |
| [`tools.build_level_pack`](#toolsbuild_level_pack) | generate, verify and save a pack of levels |
| [`tools.pcg_report`](#toolspcg_report) | expressive-range report of the generator |
| [`tools.benchmark`](#toolsbenchmark) | engine throughput |
| `rl.evaluate`, `rl.ppo` | baselines and training, see [Reinforcement learning](reinforcement-learning.md) |

## `play.py`

```bash
python play.py
python play.py --level 2
python play.py --level levels/level_003.json --scale 4
python play.py --level 7 --view follow --scale 4
```

```text
usage: play.py [-h] [--level LEVEL] [--scale SCALE] [--view {full,follow}]
  --level LEVEL         level number (1-9) or JSON path
  --scale SCALE         integer window scale (default: auto-fit to 1280x800)
  --view {full,follow}  view mode: full (default) or follow
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

One simulation step is taken per frame at 60 FPS. Winning or dying freezes the episode until reset or level selection. `--view full` (the default) shows the whole world in a window the size of the world; with no `--scale`, the window is sized to fit within 1280×800.

## `tools.level_loader`

```bash
python -m tools.level_loader levels/level_001.json
```

Loads the file with `Level.from_json` and prints `OK <path>: <name>, WxH, N platforms, M hazards`. Exit code `0` if valid, `1` on `ValueError` (message on stderr).

## `tools.solve_level`

```bash
python -m tools.solve_level levels/level_007.json --out levels/solutions/level_007.json
```

```text
usage: solve_level.py [-h] [--out OUT] [--limit LIMIT] [--k K] [--no-guided] level
  level          path to a level JSON file
  --out OUT      write the solution here (default: stdout)
  --limit LIMIT  max expanded nodes (default 3,000,000)
  --k K          hold each action for k steps (default 3)
  --no-guided    ignore metadata.waypoints even when present
```

- A **greedy best-first** solver that drives the **real engine** through `save_state`/`load_state` with `max_steps=None`. It imports neither `rendering/` nor `play.py`.
- **Guided mode** (the default): if the level carries `metadata.waypoints`, they are used as sub-goals, making solving almost linear in the critical-path length (1920×1280 worlds solve in milliseconds). `--no-guided` runs a blind search.
- `k=3` steps per action lets it hit take-off windows of 2–3 frames.
- `solve(level, limit=3_000_000, k=3, guided=True, start_state=None)` returns `(sequence, nodes)`, or `(None, nodes)` when unsolved. `start_state` allows starting mid-episode.
- Output: JSON `{level, format, total_steps, gem_collected_at_step, actions (RLE)}`, described in [Levels](levels.md#reference-solutions).
- Exit code `0` if a solution was found, `1` if not (message on stderr).

## `tools.build_level_pack`

```bash
python -m tools.build_level_pack --preset small --start-seed 0 --num 1000 --out levels/packs/small_train.jsonl
```

```text
usage: build_level_pack.py [-h] --preset {huge,large,small,tall,tiny,wide}
                           [--start-seed START_SEED] --num NUM --out OUT [--jobs JOBS]
```

Generates, **verifies** (guided solver) and saves a JSON Lines pack. An unsolvable level is a bug in the generator or engine, not something to drop silently: the tool fails. `--jobs` parallelises across seed ranges with `multiprocessing`. Each line gets `metadata.stats.solver_steps`. Load a pack with `LevelPack`.

## `tools.pcg_report`

```bash
python -m tools.pcg_report --preset small --num 1000
```

```text
usage: pcg_report.py [-h] --preset {huge,large,small,tall,tiny,wide}
                     [--start-seed START_SEED] [--num NUM] [--verify]
                     [--pairs PAIRS] [--out OUT] [--jobs JOBS]
```

An expressive-range report: mean/min/max of the generator metrics per preset, as text plus CSV, and pairwise diversity (mean Hamming distance over a grid of 5×5-tile blocks). It detects collapse (all levels identical) and bias. Example output for `--preset small --num 200`:

```text
preset=small n=200
  path_ledges: mean=9.43 min=5.00 max=17.00
  rooms_on_path: mean=3.00
  min_window: mean=8.88   mean_window: mean=17.40
  linearity: mean=0.86    leniency: mean=0.02   density: mean=0.07
  backtrack_ratio: mean=1.00
  diversity (Hamming over 5x5 blocks, 200 pairs): mean=25.32
```

## `tools.benchmark`

```bash
python -m tools.benchmark --steps 100000
```

Measures **actual transitions**, including reset and `StepResult` construction, with random actions pre-generated before the stopwatch starts; it never counts terminal no-ops. It prints JSON (`steps_per_second` per level) with the Python version and platform. See [Testing and performance](testing.md#performance) for numbers.
