#!/usr/bin/env python3
"""Generate, verify and save a level pack in JSON Lines.

Each line is one level with ``metadata.stats.solver_steps`` appended. An
unsolvable generated level is a generator/engine bug, not a silent discard:
it raises and fails the run. Levels are independent, so seed ranges can be
built in parallel.

Usage:
    python -m tools.build_level_pack --preset small --start-seed 0 --num 10000 \\
        --out levels/packs/small_train.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

from pcg import PRESETS, generate_level, level_to_dict
from tools.solve_level import solve


def build_one(seed: int, preset: str) -> dict:
    config = PRESETS[preset]
    level = generate_level(seed, config)
    sequence, _ = solve(level, k=3)
    if sequence is None:
        raise RuntimeError(
            f'seed {seed} ({preset}): generator produced an unsolvable level '
            f'(a generator/engine bug, not a discard)')
    data = level_to_dict(level)
    data['metadata']['stats']['solver_steps'] = len(sequence)
    return data


def _worker(args):
    seed, preset = args
    return build_one(seed, preset)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', required=True, choices=sorted(PRESETS))
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--num', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--jobs', type=int, default=1,
                        help='parallel workers over seed ranges (default 1)')
    args = parser.parse_args(argv)

    if args.num < 1:
        parser.error('--num must be >= 1')
    seeds = range(args.start_seed, args.start_seed + args.num)

    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            results = pool.imap(_worker, ((s, args.preset) for s in seeds), chunksize=1)
            data = list(results)
    else:
        data = [build_one(s, args.preset) for s in seeds]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fh:
        for item in data:
            fh.write(json.dumps(item, separators=(',', ':')) + '\n')
    print(f'{out}: wrote {len(data)} verified levels', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
