"""Evaluate a policy on GemEnv and report success rate, return, length and speed.

Built-in policies: ``random`` (uniform over the 6 actions) and ``right``
(hold RIGHT_JUMP), the two floor baselines any learner must beat.

    python -m rl.evaluate --levels 1-9 --episodes 50
    python -m rl.evaluate --preset small --start-seed 1000000 --num-levels 100
    python -m rl.evaluate --levels 7 --obs-mode pixels --episodes 5
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .env import GemEnv


def _parse_levels(text: str) -> list[int]:
    out = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def make_policy(name: str, rng: np.random.Generator):
    if name == 'random':
        return lambda obs: int(rng.integers(6))
    if name == 'right':
        return lambda obs: 5
    raise ValueError(f'unknown policy {name!r}')


def evaluate(env: GemEnv, policy, episodes: int, seed: int, levels=None) -> dict:
    """Run ``episodes`` episodes. With ``levels`` (a list), episode ``i`` plays
    ``levels[i % len(levels)]``, so every level gets the same share."""
    returns, lengths, wins, gems = [], [], [], []
    env_steps = 0
    start = time.perf_counter()
    for ep in range(episodes):
        options = None if levels is None else {'level': levels[ep % len(levels)]}
        obs, info = env.reset(seed=seed + ep, options=options)
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, info = env.step(policy(obs))
            total += reward
            env_steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(info['step_count'])
        wins.append(info['is_success'])
        gems.append(info['has_gem'])
    elapsed = time.perf_counter() - start
    return {
        'episodes': episodes,
        'success_rate': float(np.mean(wins)),
        'gem_rate': float(np.mean(gems)),
        'mean_return': float(np.mean(returns)),
        'mean_engine_steps': float(np.mean(lengths)),
        'env_steps_per_s': env_steps / elapsed,
        'engine_steps_per_s': sum(lengths) / elapsed,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--levels', default=None, help='built-in levels, e.g. 1-9 or 1,4,7')
    src.add_argument('--preset', default=None, help='pcg preset (tiny, small, wide, ...)')
    ap.add_argument('--start-seed', type=int, default=0)
    ap.add_argument('--num-levels', type=int, default=100)
    ap.add_argument('--policy', default='random', choices=('random', 'right'))
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--obs-mode', default='grid', choices=('grid', 'pixels'))
    ap.add_argument('--view', default='follow', choices=('follow', 'full_small'))
    ap.add_argument('--frame-skip', type=int, default=4)
    ap.add_argument('--per-level', action='store_true', help='report each built-in level')
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    policy = make_policy(args.policy, rng)
    common = dict(obs_mode=args.obs_mode, view=args.view, frame_skip=args.frame_skip)
    report = {'policy': args.policy, **common}

    if args.preset is not None:
        from pcg import PRESETS, LevelDistribution
        dist = LevelDistribution(PRESETS[args.preset], args.start_seed, args.num_levels)
        env = GemEnv(levels=dist, **common)
        report['levels'] = (f'{args.preset} seeds [{args.start_seed}, '
                            f'{args.start_seed + args.num_levels})')
        report['result'] = evaluate(env, policy, args.episodes, args.seed)
    else:
        nums = _parse_levels(args.levels or '1-9')
        env = GemEnv(levels=nums, **common)
        report['levels'] = nums
        if args.per_level:
            report['result'] = {n: evaluate(env, policy, args.episodes, args.seed, [n])
                                for n in nums}
        else:
            report['result'] = evaluate(env, policy, args.episodes * len(nums),
                                        args.seed, nums)
    env.close()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
