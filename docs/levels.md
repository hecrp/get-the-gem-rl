# Levels

Levels are JSON files loaded with `Level.from_json(path)` (or `Level.from_dict(data)` for code-generated ones). Rectangles are `[x, y, width, height]`. Invalid levels raise `ValueError`.

Validate a level from the command line:

```bash
python -m tools.level_loader levels/level_001.json
```

## Format versions

The three versions coexist and `Level.from_json` accepts all of them.

| Version | Geometry | Implicit `win_condition` | `exit` |
| --- | --- | --- | --- |
| 1 | explicit rectangles (`platforms`) | `gem` (no door) | absent |
| 2 | explicit rectangles | `exit_with_gem` | `[x, y]` |
| 3 | `tiles` (character grid) | `exit_with_gem` (or `gem`) | `D` marker in the tiles |

### Version 2: explicit rectangles

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

### Version 3: tile grid

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

Sizes are multiples of 8 between 240×160 and 1920×1280.

### Fields by version

| Field | v1 | v2 | v3 | Type |
| --- | --- | --- | --- | --- |
| `version` | yes | yes | yes | `int` (1/2/3) |
| `name` | opt. | opt. | opt. | `str` |
| `width` | yes | yes | opt. (must match the tiles) | `int` |
| `height` | yes | yes | opt. | `int` |
| `player_spawn` | yes | yes | opt. (or `P` marker) | `[x, y]` |
| `gem` | yes | yes | opt. (or `G` marker) | `[x, y]` |
| `platforms` | yes | yes | opt. (added to the tiles) | `[[x,y,w,h], …]` |
| `hazards` | opt. | opt. | opt. (added to the tiles) | `[[x,y,w,h], …]` |
| `exit` | — | yes | opt. (or `D` marker) | `[x, y]` or `null` |
| `win_condition` | — | yes | opt. | `"gem"` / `"exit_with_gem"` |
| `tiles` | — | — | yes | `[str, …]` |
| `tile_size` | — | — | opt. (only `8`) | `int` |
| `max_steps` | opt. | opt. | opt. | `int \| null` |
| `metadata` | opt. | opt. | opt. | object |
| *unknown keys* | — | — | — | kept in `Level.extras` |

`win_condition` is `"gem"` or `"exit_with_gem"`; when omitted it is `exit_with_gem` if there is a door and `gem` otherwise.

### Tile characters (v3)

| Character | Meaning | Resulting entity |
| --- | --- | --- |
| `.` | air | — |
| `#` | solid | platform (after fusion) |
| `^` | spikes | hazard (4 px tall, after fusion) |
| `P` | player spawn | `player_spawn = (c*8, (r+1)*8 - 16)` |
| `G` | gem | `gem = (c*8, r*8)` |
| `D` | door | `exit = (c*8, (r+1)*8 - 16)` |

Rules: all rows have the same length; there is at most **one** `P`, `G` and `D` marker; an unknown character raises `ValueError`; a marker that is also given explicitly in the JSON is an error ("cannot be in both").

### Solid fusion

Adjacent `#` tiles fuse into rectangles, so the physics is unchanged. Per row, consecutive columns form spans `[x, y, w, 8]`; a span merges with the rectangle directly above when `x` and `w` match. For example, the 181 solid tiles of `level_007` reduce to 9 platforms. `^` tiles fuse horizontally into hazards `[x, r*8+4, 8*count, 4]`. A test checks that fused and unfused maps produce identical trajectories.

### `metadata`

A free-form JSON object. The generator writes:

- `metadata.waypoints`: a list `[[r, c0, c1], …]` of the critical path's ledges. **It is the level's solution; never expose it to an agent.**
- `metadata.rooms`: `{"grid": [cols, rows], "path": […], "tree": [[n, p], …]}`.
- `metadata.stats`: `path_ledges`, `distractors`, `min_window`, `mean_window`, `rooms_on_path` (and `solver_steps` in packs).
- `metadata.generator`: `version`, `seed`, `attempt`, `config`.

`level.metadata` and `level.extras` return deep copies.

## Built-in levels

Nine hand-made levels live in `levels/`. Each has a reference solution in `levels/solutions/`.

| File | Name | Ver. | Size (px / tiles) | `win_condition` | `max_steps` | Reference solution (steps / gem at) |
| --- | --- | --- | --- | --- | --- | --- |
| `level_001.json` | Walk | 1 | 240×160 | `gem` | — | 126 / 126 |
| `level_002.json` | Steps | 1 | 240×160 | `gem` | — | 127 / 127 |
| `level_003.json` | Pit | 1 | 240×160 | `gem` | — | 131 / 131 |
| `level_004.json` | There and back | 2 | 240×160 | `exit_with_gem` | — | 220 / 97 |
| `level_005.json` | Up and down | 2 | 240×160 | `exit_with_gem` | — | 248 / 127 |
| `level_006.json` | Pit, twice | 2 | 240×160 | `exit_with_gem` | — | 266 / 131 |
| `level_007.json` | First big one | 3 | 480×320 (60×40) | `exit_with_gem` | 3600 | 766 / 689 |
| `level_008.json` | Long haul | 3 | 720×160 (90×20) | `exit_with_gem` | 1400 | 648 / 217 |
| `level_009.json` | Sky tower | 3 | 240×480 (30×60) | `exit_with_gem` | 2700 | 1332 / 1028 |

A missing `max_steps` means the engine default of 1800. Levels 7–9 set `max_steps` to at least twice their reference solution.

## Reference solutions

`levels/solutions/level_001.json` … `level_009.json` were produced by the blind solver (`tools/solve_level.py`, `k=3`). Format:

```json
{"level": "level_007.json", "format": "rle [action, repeat]",
 "total_steps": 766, "gem_collected_at_step": 689, "actions": [[5, 4], [2, 12], …]}
```

`actions` is a run-length encoding of action integers (0–5): each `[action, repeats]` pair means "hold `action` for `repeats` steps". To replay one, expand the encoding into a flat sequence and call `game.step(a)` in order with `max_steps=None`.

## Packs and seed convention

`tools/build_level_pack.py` writes a JSON Lines file, one level per line, with `metadata.stats.solver_steps` added and `metadata.generator.seed` identifying the seed. `LevelPack` loads such a file with the same interface as `LevelDistribution`, so a benchmark can be pinned against future generator versions: a pack is regenerated, not reused, when the generator changes.

A **split** is a contiguous seed range `[start_seed, start_seed + num)`. By convention `[0, 10⁶)` is for training and `[10⁶, 2·10⁶)` for testing. See [Procedural generation](procedural-generation.md).
