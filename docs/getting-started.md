# Getting started

## Requirements

Python **3.11 or later**. The engine (`game/`) and the level generator (`pcg/`) need no third-party packages. Rendering and the RL layer use NumPy, and the PPO baseline uses PyTorch.

## Installation

```bash
git clone https://github.com/hecrp/get-the-gem-rl.git
cd get-the-gem-rl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt           # engine, renderer, Gymnasium env
python -m pip install -r requirements-train.txt     # + PyTorch for the PPO baseline
```

With `uv`:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-train.txt
```

`pygame-ce` supplies the `pygame` module; do not install both `pygame` and `pygame-ce` in the same environment.

| File | Installs |
| --- | --- |
| `requirements.txt` | `pygame-ce`, `numpy`, `gymnasium`, `pytest` |
| `requirements-train.txt` | everything above plus `torch` (PPO baseline only) |

## First steps

Play a level yourself (see [Tools](tools.md#playpy) for the controls):

```bash
python play.py --level 7 --view follow --scale 4
```

Run the engine from Python:

```python
from game import Action, Game

world = Game.from_level("levels/level_001.json", seed=42)
world.reset()
while not world.is_done:
    result = world.step(Action.RIGHT)

assert result.won
```

Create a Gymnasium environment:

```python
from rl import GemEnv

env = GemEnv(levels=[1, 2, 3, 4, 5, 6])      # built-in levels 1-6
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

Measure a random-policy baseline and train PPO:

```bash
python -m rl.evaluate --levels 1-9 --per-level --episodes 30
python -m rl.ppo --levels 1-6 --distance-shaping 1.0 --total-steps 1000000
```

Watch runs and replays in the browser (needs `requirements-dashboard.txt`; `--demo` works without PyTorch or runs):

```bash
python -m pip install -r requirements-dashboard.txt
python -m dashboard            # http://127.0.0.1:8765
python -m dashboard --demo
```

Run the tests:

```bash
python -m pytest -q -m "not slow"      # fast suite
```

## Where to go next

- How the packages fit together: [Architecture](architecture.md).
- The simulation rules and API: [Engine](engine.md).
- Training agents: [Reinforcement learning](reinforcement-learning.md).
- Watching and operating runs: [Dashboard](dashboard.md).
- Generating or writing levels: [Levels](levels.md) and [Procedural generation](procedural-generation.md).
