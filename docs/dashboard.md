# Dashboard (`dashboard/`)

A local web interface to **operate** and **observe** RL runs: a wall of tiles that replays episodes, plots training curves with confidence intervals, shows generalisation, sweeps and heat maps, and starts, stops and resumes training from the browser.

```bash
python -m pip install -r requirements-dashboard.txt     # fastapi, uvicorn (+ httpx for tests)
python -m dashboard                                    # http://127.0.0.1:8765
python -m dashboard --demo                             # synthetic data, no PyTorch needed
```

The server reads `runs/` (the file system is the only source of truth; there is no database) and never imports PyTorch: everything that needs the model runs as a subprocess (`python -m rl.ppo`, `python -m rl.rollout`).

## Contents

- [Starting it](#starting-it)
- [Replays](#replays)
- [The interface](#the-interface)
- [Recording replays and sweeps](#recording-replays-and-sweeps)
- [Observable training](#observable-training)
- [Statistics](#statistics)
- [HTTP API](#http-api)
- [Data formats](#data-formats)
- [Engine and RL extensions](#engine-and-rl-extensions)
- [Security](#security)
- [Performance](#performance)
- [Limitations](#limitations)

## Starting it

| Flag | Default | Meaning |
| --- | --- | --- |
| `--runs-dir` | `runs/` | folder of runs to serve |
| `--host` | `127.0.0.1` | loopback only (`localhost`, `::1` also accepted; anything else is refused) |
| `--port` | `8765` | |
| `--demo` | off | build `<runs-dir>/_demo/` and serve it instead (see below) |
| `--train-python` | auto | interpreter for jobs: this one if it has PyTorch and Gymnasium, else `.venv/bin/python` |
| `--no-jobs` | off | read-only: disable training/recording |

The dashboard needs Python ≥ 3.11 and the packages in `requirements-dashboard.txt`. The server process only imports `game`, `pcg` and `rendering.palette` (no pygame, NumPy or PyTorch). Jobs need the training environment (`requirements-train.txt`); without it, the **Operate** controls are disabled with an explanation and everything else works.

**Demo mode** (`--demo`) creates `runs/_demo/` without PyTorch: the nine reference solutions, a scripted policy (hold RIGHT_JUMP) and a uniform random policy replayed on the built-in levels and 200 generated `tiny` levels, two sweeps, and synthetic training curves for a three-seed group and a memorising run. Every view shows a **DEMO DATA** badge.

## Replays

A replay is not a video: it is **a level plus a sequence of actions**. The engine is deterministic, so the server rebuilds the exact trajectory with `game.replay.simulate` (about 10 ms for an 1 800-step episode) and the browser draws it on a `<canvas>`. Replays are a few kilobytes, exact, and independent of pygame.

- Replays are **numbered per run** (`#0412`, id `run/0412`): monotonic, starting at 1, never reused even if replays are deleted. Two writers (a training run and a manual recording) get distinct numbers thanks to a file lock.
- The virtual run `_reference` holds one replay per `levels/solutions/level_NNN.json` (`_reference/level_007`); they are the ghosts of the level viewer.
- Each replay stores its level's **content hash** and the level itself goes into `runs/_levels/<sha256>.json`. If the generator ever produces a different level for the same seed, the stored snapshot is used and the player shows **⚠ derived** (`drift`).
- Every replay says where its actions came from: `source.kind` = `checkpoint` (with `update` and `env_steps`), `final`, `reference`, `scripted` or `random`, and `policy` = `greedy`, `sampled`, `reference`, … Sampled actions are stored, so a replay never needs the network again.

## The interface

The top bar has **Wall** · **Runs** · **Replays** · **Levels** · **Operate**, a live indicator (`● 2 active runs`, `SSE connected`), a theme toggle (dark by default, light by `prefers-color-scheme` or toggle) and a keyboard help sheet (`?`).

### The wall

A grid (1×1, 2×1, 2×2, 3×2, 3×3, 4×3, 4×4) of tiles; a tile may span 2×1, 1×2 or 2×2. Each tile has settings (⚙), maximise (⤢, `F` or double-click; `Esc` returns) and close (✕); drag a tile's header onto another to swap them. A layout is versioned JSON (`{"v":1,"cols":4,"rows":3,"tiles":[...]}`), saved by name in the browser, exported/imported as a file and **shared as a URL** (compressed into the hash). Three factory layouts: **Monitor de entrenamiento** (training monitor), **Muro de replays** (replay wall) and **Comparar runs** (compare runs). The interface is in Spanish. In tile settings, `$latest` means the most recently active run and `$all` every run.

| Tile | Shows |
| --- | --- |
| `replay` | one replay; camera, speed, ghosts, loop, trail |
| `replay-feed` | a queue of replays from a run/filter, played one after another; `follow` adds new replays live |
| `replay-grid` | up to 16 players, optionally on one shared clock |
| `ghosts` | several replays of the same level overlaid, with a legend (e.g. checkpoints plus the reference) |
| `curves` | training curves of runs and groups; x = env steps, time or update; log scale; EMA smoothing (the raw series drawn faintly behind); group bands |
| `train-vs-test` | success on training vs unseen levels with Wilson intervals and the generalisation gap (Newcombe) |
| `run-card` | state, progress, SPS, ETA, config, last success, and **Stop** / **Resume** / **Record replays** |
| `runs-table` | sortable, filterable runs with sparklines |
| `level-success` | success per built-in level with intervals and `n` |
| `difficulty` | success vs `min_window`, area or `leniency` from a sweep |
| `end-reasons` | won / spikes / fall / timeout proportions |
| `heatmap` | visits or deaths per 8 px cell over one level's real geometry |
| `histogram` | episode length, steps to the gem, gem → door, actions |
| `level-viewer` | a level with its metrics; built-in levels play their reference solution |
| `job-log` | live output of a job, with a filter and download |
| `note` | a little Markdown |

### The replay player

Header: `#0412`, run, result with icon (`✔ victoria`, `✘ pinchos`, `✘ vacío`, `⏱ timeout`), policy, source (`checkpoint @ 1,23 M pasos`), level (`tiny · seed 1 000 007`), engine steps and `⚠ derivado` when drifted. The canvas uses the palette served by `/api/theme` and the renderer's pixel rules, so a frame is **pixel-identical** to `PygameRenderer.render_rgb` (verified on 24 frames of three levels, `full` and `follow`). Controls: play/pause, step (`,` `.`), speed 0.25×–16× (1× = 60 engine steps/s), scrub bar with gem/door/death markers, loop, previous/next (`n`/`p`), camera (`full` or `follow`; automatic is `full` when the world fits in 480×320), an optional trail, and an action strip with one symbol per env step. `Space` pauses everything, `+`/`−` change the global speed. Off-screen players pause and at most 16 animate at once; the rest show their last frame.

**Camera `follow`** is computed by the server with `game.camera.compute_viewport` (`?cam=follow`), so there is no second camera implementation in JavaScript.

### Pages

- **Runs**: runs and experiment groups; a run page has its config, curves, evaluation, checkpoints, replays, sweeps and job log.
- **Replays**: a filterable, paginated gallery with first/last-frame thumbnails; **Send to wall** for one replay or a selection.
- **Levels**: the nine built-in levels with their solutions, a generator explorer (preset, seed, previous/next, advanced config) and an expressive-range scatter over N seeds.
- **Operate**: forms to train, record replays and sweep (every flag of the whitelist, with ranges), the job queue and logs.

## Recording replays and sweeps

`rl/rollout.py` plays a checkpoint on levels chosen by the caller (no autoreset), several environments side by side with one batched network call per step, and checks every episode against `game.replay.simulate`.

```bash
python -m rl.rollout --run tiny_n100_shaped_s1 --preset tiny --start-seed 1000000 --num 16
python -m rl.rollout --run levels1-6_s1 --levels 1-9 --sampled
python -m rl.rollout --run tiny_n10_shaped_s1 --mode sweep --tag test1k \
    --preset tiny --start-seed 1000000 --num 1000 --workers 8
```

| Flag | Meaning |
| --- | --- |
| `--run`, `--runs-dir`, `--ckpt` | the run, and `model.pt` (default) or `ckpt/update_NNNNNN.pt` |
| `--levels 1-6` / `--preset P --start-seed S --num N` | the levels (default: the run's own setting, on held-out seeds for pcg runs) |
| `--repeat R` | episodes per level |
| `--greedy` (default) / `--sampled` | argmax or sampled actions (sampling uses a private seeded generator) |
| `--mode replays` (default) / `--mode sweep` | numbered replays, or a sweep `sweeps/<tag>.jsonl` |
| `--keep-replays K` | sweep: full replays kept for the first `K` episodes plus the best and the worst |
| `--workers N`, `--envs E`, `--device` | parallelism (CPU inference scales with processes, not threads) |
| `--verify` / `--no-verify` | re-simulate every episode (default on for replays) |

It exits with a non-zero code and a message on any error, and prints a JSON summary at the end.

## Observable training

`rl.ppo` always writes two new files that older readers ignore, and gains opt-in flags; **without flags its numbers are bit-identical to before** (checked against a `progress.csv` recorded with the previous code):

| File / flag | Meaning |
| --- | --- |
| `status.json` | `state` (`running`/`finished`/`failed`/`stopped`), `pid`, heartbeat `updated_at`, progress, SPS, device, git commit, `error`, `resumed_from`; written atomically at start, at every log and at the end |
| `metrics.jsonl` | one line per log with the `progress.csv` columns plus `t`, `lr`, `ep_len_mean`, `action_hist`, `time_left_frac` |
| `--ckpt-every U` | resumable checkpoints `ckpt/update_NNNNNN.pt` (model, optimiser, counters, config, RNG states); `--ckpt-keep K` keeps the last `K`, plus one every `--ckpt-thin T` |
| `--replay-every U` | numbered replays of the current policy every `U` updates (`--replay-n`, half greedy, half sampled; training and, for pcg, held-out levels). If recording costs more than 3 % of training time, `--replay-n` is halved and the log says so |
| `--resume [CKPT]` | resume from a checkpoint (default: the run's latest); `progress.csv`/`metrics.jsonl` are appended. Operational flags given explicitly (`--total-steps`, `--device`, …) override the checkpoint's |
| `--group NAME` | experiment name; runs of a group differ only in `--seed` |
| `--runs-dir` | parent folder of the run (default `runs/`) |
| `SIGTERM` / `SIGINT` | finish the current update, save a checkpoint, `model.pt` and a partial `eval.json` (`"partial": true`), set `state: "stopped"` and exit 0. A second signal exits immediately |

A run whose `status.json` says `running` is shown as **dead** if its `pid` no longer exists or its heartbeat is older than three observed log intervals (at least 120 s). Runs started outside the dashboard are adopted as jobs if their process is alive.

## Statistics

Every proportion is shown with its sample size `n` and a 95 % interval. Server-side, standard library only (`dashboard/stats.py`):

| Quantity | Definition |
| --- | --- |
| success, gem rate | Wilson score interval |
| generalisation gap | `success_train − success_test`, Newcombe hybrid-score interval |
| end reasons | `won`, `hazard`, `fall`, `timeout` (from `Game.end_reason`); proportions summing to 1 |
| success vs difficulty | integer buckets of `min_window`; quantile buckets of area and `leniency`; buckets with fewer than 20 episodes are marked ⚠ |
| time to gem, gem → door, episode length | median and quartiles; histograms |
| visit map | engine steps of the hitbox centre per 8×8 cell, per episode (re-simulated from the sweep's stored actions) |
| death map | `death_xy` per 8×8 cell, split by cause |
| group band | runs aligned on a common `env_steps` grid (linear interpolation), mean and 95 % bootstrap interval (1 000 resamples, fixed seed) with ≥ 3 runs; fewer runs are drawn individually with a warning |
| curves | LTTB decimation on the server (`max_points`, default 1 500); EMA smoothing in the browser |

A group is `--group`, or a hash of the configuration without `seed`/`name` and operational flags.

## HTTP API

JSON unless noted; errors are `{"error": "<code>", "detail": "…"}`. Resources that change carry `ETag` and `Cache-Control: no-cache`.

| Route | Returns |
| --- | --- |
| `GET /api/health` | version, `runs_dir`, whether jobs can use PyTorch, start time, demo mode |
| `GET /api/theme` | `rendering.palette.as_dict()` |
| `GET /api/runs`, `GET /api/runs/{name}` | runs with state (dead detection), summary config, last metrics, evaluation with Wilson intervals, replay/sweep counts; the detail adds the full config and files |
| `GET /api/runs/{name}/metrics?cols=&since=&max_points=&x=` | series decimated with LTTB; `since` = last known update |
| `GET /api/groups?cols=` | experiment groups and, with ≥ 3 seeds, mean and band per column |
| `GET /api/replays?run=&split=&policy=&outcome=&level=&source=&tag=&after=&limit=&order=&seed=` | the replay index without actions, paginated by `after` (an id), with `total`; `level` accepts a hash, `builtin:7` or `pcg:tiny:1000003`; `order` = `number`, `latest` or `random` |
| `GET /api/replays/{run}/{n}` | one replay with its actions (`_reference/level_007` for references) |
| `GET /api/replays/{run}/{n}/trajectory?cam=none\|follow` | the trajectory, gzip when accepted |
| `GET /api/levels/builtin`, `/api/levels/by-ref?ref=`, `/api/levels/by-hash/{hash}`, `/api/levels/{n}/reference` | levels and reference replays |
| `GET /api/levels/expressive?preset=&start=&n=` | `pcg.metrics` over N seeds (cached) |
| `GET /api/sweeps/{run}`, `/api/sweeps/{run}/{tag}/summary`, `/api/sweeps/{run}/{tag}/heatmap?level=&layer=` | sweeps, their aggregates and heat maps |
| `GET /api/jobs`, `/api/jobs/schema`, `/api/jobs/{id}`, `/api/jobs/{id}/log?tail=` | jobs, the flag whitelist, a job, its log (text) |
| `POST /api/jobs` | `{kind: "train"\|"rollout"\|"sweep", args: {...}, then?: [...]}` → `202` |
| `DELETE /api/jobs/{id}` | `SIGTERM` (clean stop); `?force=1`, 30 s later, `SIGKILL` |
| `GET /api/events` | server-sent events: `run.updated`, `job.updated`, `replay.added`, `heartbeat` every 15 s |

Limits: trajectory responses up to 8 MB (else `413`), `limit` up to 500, request bodies up to 64 KB.

## Data formats

**`LevelRef`** (also `info["level_ref"]` in `GemEnv`):

```json
{"kind": "builtin", "n": 7}
{"kind": "pcg", "preset": "tiny", "seed": 1000007, "config": {"cols": 30, "rows": 20, "...": "..."}, "generator_version": 1}
{"kind": "custom", "hash": "sha256:…"}
```

`config` is the full `GenConfig`, since presets may change. Levels from a `LevelPack` are `custom` (they are frozen by content) with `pack` and `index`.

**Level store**: `runs/_levels/<sha256>.json` holds `Level.to_dict()` as canonical JSON (sorted keys, no spaces). `level_hash(level)` is `"sha256:" + hexdigest` of that text.

**Replay** (one line of `runs/<run>/replays.jsonl`):

```json
{"schema": 1, "n": 412, "id": "tiny_n100_shaped_s1/0412", "run": "tiny_n100_shaped_s1",
 "created": "2026-11-24T12:00:00Z",
 "source": {"kind": "checkpoint", "update": 300, "env_steps": 1228800, "ckpt": "ckpt/update_000300.pt"},
 "policy": "greedy", "split": "test",
 "level": {"ref": {"kind": "pcg", "...": "..."}, "hash": "sha256:…", "name": "gen-v1-1000003", "width": 240, "height": 160},
 "env": {"frame_skip": 4, "max_steps": 1800, "physics": null},
 "actions": [[5, 12], [3, 1], [5, 30]],
 "outcome": {"result": "won", "end_reason": "won", "engine_steps": 312, "env_steps": 78,
             "gem_step": 140, "win_step": 312, "death_xy": null, "return": 2.61},
 "tags": []}
```

`actions` is RLE over env steps; each holds its action for `frame_skip` engine steps and stops at the terminal step, as `GemEnv` does (reference solutions use `frame_skip: 1`). `created` is metadata only and never orders anything.

**Trajectory** (API only): `xy` = `2·(N+1)` positions (index 0 = after reset), `flags` = `N+1` integers (bit 0 grounded, 1 gem collected, 2 dead, 3 won, 4 exit open), `events` (`gem`, `exit_open`, `won`, `dead` with `cause` and the `hazard` index, `timeout`), `cam` with `?cam=follow`, plus `drift` and `level_hash`.

**Sweep episode** (one line of `runs/<run>/sweeps/<tag>.jsonl`): `i`, `level_ref`, `level_hash`, `policy`, `success`, `gem`, `engine_steps`, `env_steps`, `end_reason`, `gem_step`, `win_step`, `death_xy`, `level_metrics` (`pcg.metrics` for generated levels; size for built-in ones), `action_hist`, plus `split`, `return`, `frame_skip` and the episode's RLE `actions` (so heat maps can be rebuilt). `<tag>.meta.json` records the checkpoint, policy, seed range, generator version and `n`.

**Readers and writers**: JSON files are written to a temporary file and `os.replace`d; JSON Lines get one `write` per newline-terminated line; readers ignore a trailing line without `\n` and flag (never hide) a complete line that is not valid JSON.

## Engine and RL extensions

Phase 5 extended the core only additively; physics, actions, level formats, the generator and its output are unchanged (`GENERATOR_VERSION` and the golden hashes are untouched), and the benchmark is unchanged within noise.

| Where | Addition |
| --- | --- |
| `game/level.py` | `Level.to_dict()` (exact round trip through `Level.from_dict`), `level_hash(level)`, `canonical_json(obj)` |
| `game/game.py` | `Game.end_reason`: `None`, `'won'`, `'hazard'`, `'fall'` or `'timeout'`, derived from existing state (precedence dead > won > timeout) |
| `game/replay.py` | `rle_encode`, `rle_decode`, `simulate(level, actions, frame_skip=, max_steps=, physics=, record=)` → `Trajectory`, `reference_replays(solutions_dir)` |
| `game/replay_store.py` | atomic JSON writes, tolerant JSON Lines reader, the level store, locked replay numbering |
| `rendering/palette.py` | colours and shape rules as standard-library data; `PygameRenderer` uses them (output unchanged); `rendering` now imports pygame lazily |
| `pcg/__init__.py` | `LevelDistribution.last_ref` / `LevelPack.last_ref`, `pcg_ref`, `preset_name`, `LevelDistribution(..., preset=)` |
| `rl/env.py` | `info["level_ref"]` on every `reset` and `step`; `GemEnv.level_ref`; `rl.env.level_ref(hint, level)` |
| `rl/rollout.py` | new: replays and sweeps from a checkpoint |
| `rl/ppo.py` | `status.json`, `metrics.jsonl`, signals, `--ckpt-every`, `--replay-every`, `--resume`, `--group`, `--runs-dir` |

Public contract changes: `info` of `GemEnv` gains `level_ref`; runs gain `status.json` and `metrics.jsonl`; `config.json` gains the new flags' keys. `progress.csv`, `eval.json` and `model.pt` keep their format.

## Security

The server binds to loopback only and rejects requests whose `Host` is not loopback (DNS rebinding). `POST`/`DELETE` need an `Origin` equal to the server itself and the `X-Dashboard-Token` printed at start-up and embedded in the served page (CSRF). Jobs are validated against a whitelist of flags with type and range (`dashboard/job_schema.py`; anything else is `422`), run names must match `[A-Za-z0-9_.-]{1,64}` and not start with `_`, an existing run is never overwritten (`409`), and jobs are started with an argument list (never a shell), the repository as working directory and a minimal environment.

## Performance

Measured on an Apple M4 Pro:

| Aspect | Measured |
| --- | --- |
| server start with 50 runs, until `GET /api/runs` | 0.38 s |
| `GET /api/runs/{name}/metrics` on 100 000 rows | 154 ms cold, 2 ms warm, 1 500 points |
| trajectory of 1 800 steps | 11 ms (simulation), 2 ms cached; under 4 KB gzip |
| `rl.rollout` on the CPU | ≈ 3 400 env steps/s per process (convolution-bound); 1 000 `tiny` episodes in 21 s with 8 workers, 2 min 9 s with one |

## Limitations

- Playing in the browser (`WS /api/play`) is not implemented.
- `--resume` restores the model, optimiser, counters and RNG states, but episodes in flight are not checkpointed: the environments restart, so a resumed run is consistent but not bit-identical to an uninterrupted one.
- Saved layouts live in the browser's `localStorage`; use **Export** or a shared URL to move them.
- The JavaScript polygon fill reproduces pygame-ce exactly for every shape the renderer draws up to 32 px tall (all spikes in the game are 4 px); taller spike strips can differ by one pixel on their bottom row.
