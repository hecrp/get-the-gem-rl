# Reinforcement learning (`rl/`)

`rl.GemEnv` is a [Gymnasium](https://gymnasium.farama.org/) environment over the engine. Rewards, observations, frame skipping and level sampling live only in `rl/`; `game/` and `pcg/` never import it (or Gymnasium). `terminated = won or dead`, `truncated = timeout`.

```python
from rl import GemEnv
from pcg import PRESETS, LevelDistribution

train = LevelDistribution(PRESETS["small"], start_seed=0, num_levels=200)
env = GemEnv(levels=train, frame_skip=4, gem_reward=0.5)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## `GemEnv`

```python
GemEnv(levels=1, *, obs_mode='grid', grid_size=(31, 21), view='follow',
       goal_vector=True, frame_skip=4, win_reward=1.0, gem_reward=0.0,
       death_penalty=0.0, step_penalty=0.0, distance_shaping=0.0,
       max_steps=FROM_LEVEL, render_mode=None, level_cache_size=64)
```

### Levels

`levels` can be a built-in number (1–9), a path, a `Level`, a list of those, or anything with `sample(rng)` (`LevelDistribution`, `LevelPack`). A level is drawn from `env.np_random` on each `reset`. `reset(options={"level": ...})` forces one, which is useful to evaluate on held-out seeds.

### Observation

With `obs_mode="grid"` (the default) the observation is a dict:

- `grid`: `(5, 21, 31)` `uint8`, an egocentric tile crop centred on the player with channels `solid, hazard, gem, door, outside`. Side walls count as solid, and the gem channel clears once the gem is carried.
- `vec`: `(12,)` `float32`: velocity, `grounded`, `has_gem`, `needs_door`, the player's offset inside its tile, the gem and door offsets in screens, and the fraction of time left. `goal_vector=False` zeroes the gem/door offsets so the agent must find them by looking.

With `obs_mode="pixels"`, `grid` is replaced by `pixels`, a `(160, 240, 3)` frame:

- `view="follow"`: camera on the player (partial observability).
- `view="full_small"`: the whole world, for worlds up to 480×320.

Both have the same shape, so the same network can compare partial and full observability. Grid mode never imports pygame. See [Rendering](rendering.md) for the views.

### Actions

`Discrete(6)`: `NOOP, LEFT, RIGHT, JUMP, LEFT_JUMP, RIGHT_JUMP`, held for `frame_skip` engine steps (default 4). The repeat stops at a terminal step, and `info["engine_steps"]` says how many ran.

### Reward

Sparse by default: `+win_reward` (1.0) on victory. Optional terms:

| Parameter | Effect |
| --- | --- |
| `gem_reward` | once, on pickup, only in door levels |
| `death_penalty` | applied on death |
| `step_penalty` | applied per engine step |
| `distance_shaping` | potential-based shaping, described below |

**Distance shaping** (`distance_shaping=c`) is potential-based shaping (Ng, Harada & Russell, 1999): `c·(Φ(s′) − Φ(s))` with Φ = minus the straight-line route still to run (player → gem → door, or player → door once carried), measured in screens. It telescopes over an episode, so it speeds learning without changing which policies are optimal. Straight lines ignore walls, so it can mislead around obstacles.

### `info`

`is_success`, `has_gem`, `gem_collected_event`, `step_count`, `level_name`, `engine_steps`, and `level_ref`: where the level came from, as a JSON-native dict (`{"kind": "builtin", "n": 7}`, `{"kind": "pcg", "preset": "tiny", "seed": …, "config": {…}, "generator_version": 1}` or `{"kind": "custom", "hash": "sha256:…"}`), on every `reset` and `step`. It comes from `LevelDistribution.last_ref`/`LevelPack.last_ref` or from the number, path or `Level` given, and never touches an RNG. The generator's `metadata` (in particular `waypoints`, the level's solution) is never exposed.

## Baselines and speed

```bash
python -m rl.evaluate --levels 1-9 --per-level --episodes 30      # uniform random policy
python -m rl.evaluate --policy right --levels 1-9 --per-level --episodes 1
python -m rl.evaluate --preset small --start-seed 1000000 --num-levels 100
```

Measured on an Apple M4 Pro (one core, `frame_skip=4`):

- A uniformly random policy wins only level 1, 10 % of the time, and none of the generated `small` levels. Holding RIGHT_JUMP wins level 3 and nothing else.
- Throughput is about 38 000 env steps/s (≈150 000 engine steps/s) in grid mode, 3 400 env steps/s with `pixels`/`follow` and 490 env steps/s with `pixels`/`full_small`.

## PPO baseline (`rl/ppo.py`)

A single-file PPO in the style of CleanRL: a small CNN reads the tile grid, an MLP the feature vector, and a shared actor-critic head sits on top. Timeouts are bootstrapped from the value of the final observation. It uses Apple-silicon `mps` (or CUDA) when available; on the CPU the convolutions are about 15× slower. It needs the packages in `requirements-train.txt`.

```bash
python -m rl.ppo --levels 1-6 --total-steps 3000000 --eval-every 50
python -m rl.ppo --levels 1-6 --total-steps 3000000 --eval-every 50 --distance-shaping 1.0
python -m rl.ppo --preset tiny --num-levels 100 --total-steps 5000000 --distance-shaping 1.0 --eval-every 100
```

Each run writes `runs/<name>/`:

| File | Contents |
| --- | --- |
| `config.json` | the run's configuration |
| `progress.csv` | training curve, plus train/test success at each evaluation |
| `eval.json` | final success per level, or on training seeds vs held-out seeds (from 1 000 000 on) |
| `model.pt` | the trained model |
| `status.json` | state (`running`, `finished`, `failed`, `stopped`), pid, heartbeat and progress |
| `metrics.jsonl` | the `progress.csv` columns plus elapsed time, learning rate, episode length, action histogram and time left |
| `ckpt/update_NNNNNN.pt` | resumable checkpoints, with `--ckpt-every` |
| `replays.jsonl` | numbered replays of the policy, with `--replay-every` or `rl.rollout` |

Opt-in flags (without them the run is bit-identical to before): `--ckpt-every`, `--ckpt-keep`, `--ckpt-thin`, `--replay-every`, `--replay-n`, `--resume [CKPT]`, `--group`, `--runs-dir`. `SIGTERM`/`SIGINT` stop after the current update with a checkpoint, `model.pt`, a partial `eval.json` and `state: "stopped"`. `python -m rl.rollout` records replays or evaluation sweeps from any checkpoint. See [Dashboard](dashboard.md#observable-training).

## First results

Apple M4 Pro, one seed per setting, 64 evaluation episodes per set; 32 envs × 128 steps, `frame_skip=4`, `gem_reward=0.5`:

| Setting | Steps | Wall time | Result |
| --- | --- | --- | --- |
| Levels 1–6, sparse reward | 3 M | 11 min | 100 % on every level. Evaluation success 2 % at 0.2 M steps, 53 % at 0.6 M, 100 % from 1.0 M. |
| Levels 1–6, `distance_shaping=1.0` | 3 M | 11 min | 100 % on every level, 97 % already at 0.2 M steps (≈5× faster). |
| `tiny`, 10 training levels, shaped | 5 M | 28 min | 100 % on training levels, **31 %** on unseen levels (20 % greedy): memorisation. |
| `tiny`, 100 training levels, shaped | 5 M | 27 min | 100 % train, 98 % unseen (83 % greedy). |
| `tiny`, 1 000 training levels, shaped | 5 M | 29 min | 100 % train, 100 % unseen. |
| `tiny`, unbounded levels, shaped | 5 M | 29 min | 100 % train, 100 % unseen. |

The results reproduce, in miniature, the Procgen/CoinRun finding that agents trained on few levels memorise them instead of generalising. A uniformly random policy wins 2 % of unseen `tiny` levels. The four `tiny` runs ran concurrently.

These are single-seed results. `tiny` is one screen at difficulty 0.2, so generalisation saturates by 100 levels; harder presets, `goal_vector=False` or no shaping should spread the curve out.

## Notes for building on `GemEnv`

- **Keep `Level` objects stable per worker.** The renderer caches its static layer by `Level` identity; `LevelDistribution.get(i)` returns the same object for a given seed.
- **Use fixed-shape views to batch pixels.** `follow` and `full_small` both give `(160, 240, 3)`; `FULL` with `canvas_size=None` changes shape with the level.
- **Never expose `metadata.waypoints`.** It is the critical path of the level, and handing it to the agent gives the route away.
- **Frames cost far more than steps.** With pixel observations, drawing dominates throughput (see [Rendering](rendering.md#frame-cost)).

## Not yet

Multiple seeds for the PPO results (the [dashboard](dashboard.md) draws bands and intervals once they exist), harder generalisation settings, pixel-based training, level editing for curricula (ACCEL-style `mutate`), and packaging as an installable library.
