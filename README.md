# Get the Gem

A small 2D platformer built to be a reinforcement-learning environment. The agent must reach a gem, carry it back to a door, and avoid spikes and pits on the way.

- **Deterministic engine** in pure standard-library Python, about 150 000 steps/s on one core.
- **Procedural level generator** whose levels are winnable by construction, with Procgen-style train/test seed splits.
- **[Gymnasium](https://gymnasium.farama.org/) environment** (`rl.GemEnv`) with tile-grid or pixel observations.
- **PPO baseline** and first results, including a small reproduction of the Procgen/CoinRun finding that agents trained on few levels memorise them instead of generalising.

## Quick start

Requires Python 3.11 or later.

```bash
git clone https://github.com/hecrp/get-the-gem-rl.git
cd get-the-gem-rl
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-train.txt
```

```bash
python play.py --level 7 --view follow --scale 4                              # play it yourself
python -m rl.evaluate --levels 1-9 --per-level --episodes 30                  # random-policy baseline
python -m rl.ppo --levels 1-6 --distance-shaping 1.0 --total-steps 1000000    # train PPO
```

## Using it from Python

```python
from rl import GemEnv
from pcg import PRESETS, LevelDistribution

train = LevelDistribution(PRESETS["small"], start_seed=0, num_levels=200)
env = GemEnv(levels=train, frame_skip=4, gem_reward=0.5)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Documentation

Full documentation lives in [`docs/`](docs/README.md):

- [Getting started](docs/getting-started.md) · [Architecture](docs/architecture.md)
- [Engine](docs/engine.md) · [Levels](docs/levels.md) · [Procedural generation](docs/procedural-generation.md) · [Rendering](docs/rendering.md)
- [Reinforcement learning](docs/reinforcement-learning.md) · [Dashboard](docs/dashboard.md) · [Tools](docs/tools.md) · [Testing and performance](docs/testing.md)

## Project layout

```text
game/        deterministic engine
pcg/         procedural level generator
rendering/   offscreen pygame renderer
rl/          Gymnasium environment, PPO and replay recording
dashboard/   local web dashboard (replays, curves, jobs)
tools/       solver, level packs, reports
levels/      nine built-in levels
docs/        documentation
tests/       pytest suite
```

## License

[MIT](LICENSE)
