# Architecture

**Get the Gem** is a deterministic pixel platformer built to be a reinforcement-learning environment. The simulation advances in fixed steps, independent of the wall clock and of rendering. The goal is a reprogrammable benchmark: procedurally generated levels that are always winnable, permanent seeds, and a simulation core with no external dependencies.

## The game

The player (12×16 px) must collect a **gem** and, in most levels, carry it back to a **door**. The player dies on touching **spikes** or falling below the world, and loses by **timeout** when the step limit runs out.

Mechanics, in simulation order:

- **Gem**: a non-solid trigger (strict AABB overlap). Collecting it sets `gem_collected` once per episode; it never respawns. In `win_condition == "gem"` levels, collecting it wins.
- **Door** (12×16 px): present in `exit_with_gem` levels. It opens (victory) only when the player touches it **while carrying the gem**.
- **Death**: touching a hazard, or `player.y > level.height` (the whole hitbox below the world).
- **Timeout**: `step_count >= max_steps` without winning or dying in that step.

See [Engine](engine.md) for the exact contract.

## Packages

| Package | Contents | Imports |
| --- | --- | --- |
| `game/` | headless engine: `Game`, `Level`, `GameState`, `StepResult`, `GameSnapshot`, `Action`, `PhysicsConfig`, collision, camera; replays (`game.replay`) and their files (`game.replay_store`) | standard library only |
| `pcg/` | procedural generator: `generate_level`, `GenConfig`, `PRESETS`, `LevelDistribution`, `LevelPack`, `JumpTable`, metrics, PRNG | `game/` + standard library (no pygame, no NumPy) |
| `rendering/` | `PygameRenderer`, `ViewConfig`, `ViewMode`; `palette` (colours and shape rules, standard library only) | `game.camera` + pygame + NumPy |
| `rl/` | Gymnasium environment `GemEnv`, evaluation script, PPO baseline, replay/sweep recorder (`rl.rollout`) | `game/` + NumPy + Gymnasium; `rendering/` only for pixel observations; PyTorch for PPO and `rl.rollout` |
| `dashboard/` | local web dashboard (FastAPI server + ES-module front end) | `game/`, `pcg/`, `rendering.palette` + FastAPI/uvicorn; never PyTorch, NumPy or pygame (jobs run `rl` in subprocesses) |
| `tools/` | solver, level validation, level packs, generator report, benchmark | `game/` + `pcg/` |
| `play.py` | interactive keyboard player | `game/` + `rendering/` + pygame |
| `levels/` | nine built-in levels and their reference solutions | — |
| `tests/` | pytest suite | depends on the module under test |

```text
game/        deterministic engine, standard library only
pcg/         procedural level generator (standard library + game)
rendering/   offscreen pygame renderer and views
rl/          Gymnasium environment, baselines, PPO and replay recording
dashboard/   local web dashboard
tools/       level validation, solver, level packs, generator report, benchmark
levels/      nine built-in levels and their reference solutions
play.py      keyboard play
tests/       pytest suite
docs/        this documentation
```

## Dependency graph

```text
game/    (standard library only)
   ↑
pcg/     (game/)                          never imports pygame or NumPy
   ↑
tools/   (game/ + pcg/)
rendering/ (game.camera + pygame + NumPy)
   ↑
rl/      (game/ + NumPy + Gymnasium; rendering/ only in pixel mode)
dashboard/ (game/ + pcg/ + rendering.palette; rl/ only as subprocesses)
play.py  (game/ + rendering/ + pygame)
```

Dependencies only point downwards:

- `game/` and `pcg/` never import `rl/`, Gymnasium, pygame or NumPy. A test loads both in a clean subprocess and checks that neither pygame nor NumPy appears.
- Only `rendering/pygame_renderer.py` and `play.py` import pygame; NumPy is used by the renderer and the RL layer. In grid mode the RL layer never imports pygame.
- `dashboard/` imports `game`, `pcg` and `rendering.palette` only; anything that needs PyTorch runs as `python -m rl.ppo` / `python -m rl.rollout` subprocesses. `rl/` never imports `dashboard/`. A test checks the import graph and that the server process never loads PyTorch, NumPy or pygame.
- **Rewards, observations, frame skipping and level sampling live only in `rl/`.** The engine reports what happened (`won`, `dead`, `timeout`) and has no notion of reward.

## Design principles

- **Determinism.** The same seed and the same actions always produce the same states, on any machine. The generator uses its own integer-only PRNG, so a `(version, seed, config)` triple always yields the same level. See [Engine – Determinism](engine.md#determinism).
- **Winnable by construction.** The generator measures what jumps are possible by simulating the real engine, builds the critical path first, and keeps it reversible. See [Procedural generation](procedural-generation.md).
- **Headless first.** Rendering never changes the simulation, and obtaining a frame never opens a window.
- **Small surface.** No framework, asset pipeline or alternative engine; the engine is under a thousand lines of standard-library Python.

## Further reading

- [Engine](engine.md) · [Levels](levels.md) · [Procedural generation](procedural-generation.md)
- [Rendering](rendering.md) · [Reinforcement learning](reinforcement-learning.md)
- [Dashboard](dashboard.md) · [Tools](tools.md) · [Testing and performance](testing.md)
