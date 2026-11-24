"""Measure actual simulation transitions, including reset and result construction.

No throughput assertion: wall-clock speed is machine-dependent. Random actions
are generated before timing; terminal no-ops are never included in the count.
"""
import argparse
import json
from pathlib import Path
import platform
import random
from time import perf_counter

from game import Action, Game

ROOT = Path(__file__).resolve().parents[1]


def benchmark(steps: int) -> dict:
    rng = random.Random(1234)
    actions = tuple(Action)
    sequence = [rng.choice(actions) for _ in range(steps)]
    rows = []
    level_paths = sorted((ROOT / 'levels').glob('level_*.json'))
    for path in level_paths:
        game = Game.from_level(path, seed=1234, max_steps=500)
        # Warmup is outside the measured region.
        for action in sequence[:min(1000, steps)]:
            if game.is_done:
                game.reset()
            game.step(action)
        game.reset(seed=1234)
        effective_steps = 0
        resets = 0
        started = perf_counter()
        for action in sequence:
            if game.is_done:
                game.reset()
                resets += 1
            before = game.step_count
            game.step(action)
            effective_steps += game.step_count - before
        elapsed = perf_counter() - started
        if effective_steps != steps:
            raise RuntimeError('Benchmark included ineffective steps')
        rows.append({
            'level': path.stem, 'steps': steps, 'effective_steps': effective_steps,
            'resets': resets, 'seconds': elapsed, 'steps_per_second': steps / elapsed,
        })
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'seed': 1234, 'includes_reset_and_result': True, 'levels': rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=100_000,
                        help='actual transitions per level (default: 100000)')
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error('--steps must be positive')
    print(json.dumps(benchmark(args.steps), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
