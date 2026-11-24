# Testing and performance

## Running the tests

```bash
python -m pytest -q                    # full suite (~3 min)
python -m pytest -q -m "not slow"      # fast suite (~25 s)
python -m pytest -m slow               # only the slow tests (solvability, large worlds)
python -m tools.benchmark --steps 100000
node --test tests/dashboard_js/        # front-end modules (optional, needs Node)
```

The dashboard tests need `requirements-dashboard.txt` and are skipped without it; the `rl` tests need PyTorch.

`pytest.ini` sets `testpaths = tests`, `pythonpath = .` and `addopts = -ra`, and registers the `slow` marker. Renderer tests use SDL's dummy display, so no window opens.

## What the suite covers

The suite covers the engine contract and determinism (including 100 000 seeded random steps across all levels with per-step invariants), exact replay of every reference solution, the generator's golden hashes and solvability, renderer output, the Gymnasium API (`check_env`), the tile grid against an independent rasterisation, and end-to-end PPO smoke runs.

| File | Covers |
| --- | --- |
| `test_game.py`, `test_core_contract.py` | lifecycle, `step` order, terminal no-op, `reset`, properties, immutability |
| `test_physics.py`, `test_collision.py`, `test_broadphase.py` | integration, strict AABB collisions, broad-phase equal to brute force |
| `test_determinism.py` | same seed ⇒ same states; 100 000 live steps |
| `test_action_types.py` | action input contract: integer scalars (incl. NumPy) accepted; `bool`, `float`, `str`, arrays rejected |
| `test_max_steps.py` | `max_steps` precedence and world-size validation |
| `test_snapshot.py` | `save_state` / `load_state` |
| `test_levels.py`, `test_tiles.py`, `test_fusion.py` | `Level` validation, formats, tile parsing, solid fusion |
| `test_exit.py`, `test_exit_levels.py`, `test_exit_determinism.py` | door mechanics, `win_condition`, death vs victory, door-level invariants |
| `test_conformity.py` | exact figures of level 7 and replay of the nine reference solutions |
| `test_solver.py` | the solver solves known levels and detects impossible ones |
| `test_camera.py` | `compute_viewport`, clamping, worlds smaller than the window |
| `test_renderer.py`, `test_renderer_view.py`, `test_exit_renderer.py` | framebuffers, FOLLOW/FULL, `downscale`, palette, door and carried-gem appearance |
| `test_pcg_rng.py` | `SplitMix64`, `fnv1a`, `substream` (golden vectors) |
| `test_pcg_jump_table.py` | measured jump table, left/right symmetry, walk-off |
| `test_pcg_generator.py` | golden hashes, metadata, solid waypoints, stdlib isolation |
| `test_pcg_engine.py` | generated levels are solvable, no soft-locks |
| `test_pcg_distribution.py` | `LevelDistribution` / `LevelPack`, seed splits, duck-typed `sample` |
| `test_rl_env.py` | Gymnasium API conformity, rewards, frame skip, egocentric grid |
| `test_rl_ppo.py` | PPO smoke run: a few tiny updates end to end on the CPU |
| `test_play.py`, `test_exit_play.py` | keyboard handling, level catalogue and key bindings, interactive loop smoke |
| `test_benchmark.py` | the benchmark measures real transitions |
| `test_replay.py`, `test_replay_env.py` | RLE, `simulate`, events, `Game.end_reason`; `simulate` ≡ `GemEnv` on 200 episodes; `info["level_ref"]` |
| `test_level_serial.py` | `Level.to_dict` round trip (9 built-in + 200 generated levels), `level_hash` stability |
| `test_palette.py` | palette values and pygame-free import |
| `test_rollout.py`, `test_readers.py` | replay recording, numbering under concurrency, `--verify`, sweeps; tolerant readers and atomic writers |
| `test_ppo_hooks.py` | `status.json`, `metrics.jsonl`, checkpoints, replays, `--resume`, `SIGTERM`, bit-exact `progress.csv` without flags |
| `test_dashboard_*.py` | the dashboard API, statistics (published reference values), security, jobs, SSE, demo mode without PyTorch, dependency rules |
| `dashboard_js/*.test.mjs` | front-end pure modules with `node --test`: layouts, clock, RLE, formatters, and the polygon fill against pixels rendered by pygame |

## Performance

Measured on an Apple M4 Pro, Python 3.12, one core.

### Simulation

Without rendering, with random actions and reset on termination:

| Level | Steps / s |
| --- | --- |
| `level_001` (240×160) | ≈ 193 000 |
| `level_007` (480×320) | ≈ 192 000 |
| generated `large` (960×640) | ≈ 159 000 |

The grounded probe in `reset()` adds about 1.9 µs per reset (once per episode), which is imperceptible per step.

### Rendering

`render_rgb` costs 0.25–2.05 ms per frame depending on the view; see [Rendering](rendering.md#frame-cost).

### Generation

`generate_level` takes from 0.7 ms (`tiny`) to 150 ms (`huge`); see [Procedural generation](procedural-generation.md#generation-speed).

### Environment

About 38 000 env steps/s in grid mode with `frame_skip=4`; see [Reinforcement learning](reinforcement-learning.md#baselines-and-speed).
