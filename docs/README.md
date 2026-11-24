# Documentation

Reference documentation for **Get the Gem**, a deterministic platformer built as a reinforcement-learning environment. For a short overview of the project, see the [main README](../README.md).

## Contents

| Document | What it covers |
| --- | --- |
| [Getting started](getting-started.md) | Requirements, installation and first commands |
| [Architecture](architecture.md) | The game, the packages, the dependency rules and the design principles |
| [Engine](engine.md) | The `game/` API and contract: actions, `Game`, state objects, physics, determinism |
| [Levels](levels.md) | JSON level format (v1–v3), tile characters, the nine built-in levels, solutions and packs |
| [Procedural generation](procedural-generation.md) | The `pcg/` generator: how it works, presets, seeds, distributions and metrics |
| [Rendering](rendering.md) | The pygame renderer: views, frame shapes, palette, cost and caching |
| [Reinforcement learning](reinforcement-learning.md) | `GemEnv`, observations, rewards, the PPO baseline and first results |
| [Dashboard](dashboard.md) | The local web dashboard: replay wall, curves with confidence intervals, sweeps, heat maps, starting and stopping training |
| [Tools](tools.md) | `play.py` and the command-line tools: solver, validator, level packs, reports, benchmark |
| [Testing and performance](testing.md) | Running the suite, what it covers, and measured throughput |

## Suggested reading

- **I want to run it**: [Getting started](getting-started.md), then [Tools](tools.md) to play a level.
- **I want to train an agent**: [Reinforcement learning](reinforcement-learning.md), then [Procedural generation](procedural-generation.md) for train/test level splits, and the [Dashboard](dashboard.md) to watch it.
- **I want to use the engine on its own**: [Architecture](architecture.md), then [Engine](engine.md).
- **I want to make levels**: [Levels](levels.md), then [Tools](tools.md) to validate and solve them.
- **I want to contribute**: [Architecture](architecture.md) and [Testing and performance](testing.md).

## Conventions

- Coordinates are in pixels, with the origin at the top-left and `y` growing downwards.
- Performance figures were measured on an Apple M4 Pro with Python 3.12 and will differ on other machines.
- Code examples assume the repository root as the working directory and an activated virtual environment.
