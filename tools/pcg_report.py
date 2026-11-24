#!/usr/bin/env python3
"""Expressive-range report.

Generates N levels and summarises the distribution of metrics (text + CSV),
plus pairwise diversity (mean Hamming distance over a 5x5-tile block grid).
Detects collapses (all levels identical), biases and whether ``difficulty``
moves the metrics in the expected direction.

Usage:
    python -m tools.pcg_report --preset small --num 1000
"""
import argparse
import csv
import random
import statistics
import sys
from pathlib import Path

from pcg import PRESETS, generate_level, metrics

FIELDS = ['path_ledges', 'rooms_on_path', 'min_window', 'mean_window',
          'linearity', 'leniency', 'density', 'backtrack_ratio']


def _block_grid(tiles, block=5):
    rows = len(tiles)
    cols = len(tiles[0]) if rows else 0
    out = []
    for r in range(0, rows, block):
        line = []
        for c in range(0, cols, block):
            solid = any('#' in tiles[rr][c:c + block] for rr in range(r, min(r + block, rows)))
            line.append('#' if solid else '.')
        out.append(''.join(line))
    return out


def _hamming(a, b):
    return sum(1 for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x != y)


def _summary(label, values, fmt='{:.2f}'):
    print(f'  {label}: mean={fmt.format(statistics.mean(values))} '
          f'min={fmt.format(min(values))} max={fmt.format(max(values))}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', required=True, choices=sorted(PRESETS))
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--pairs', type=int, default=200)
    parser.add_argument('--out', help='CSV output path (default: <preset>_report.csv)')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args(argv)

    config = PRESETS[args.preset]
    seeds = range(args.start_seed, args.start_seed + args.num)

    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            levels = list(pool.imap(lambda s: generate_level(s, config), seeds))
    else:
        levels = [generate_level(s, config) for s in seeds]

    rows = []
    for lv in levels:
        m = metrics(lv)
        row = {f: getattr(m, f) for f in FIELDS}
        if args.verify:
            from tools.solve_level import solve
            seq, _ = solve(lv, k=3)
            row['solver_steps'] = len(seq) if seq is not None else None
        rows.append(row)

    print(f'preset={args.preset} n={len(levels)}')
    for f in FIELDS:
        _summary(f, [r[f] for r in rows])
    if args.verify:
        steps = [r['solver_steps'] for r in rows if r['solver_steps'] is not None]
        if len(steps) != len(rows):
            print(f'  UNSOLVED: {len(rows) - len(steps)}/{len(rows)} levels', file=sys.stderr)
        if steps:
            _summary('solver_steps', steps, '{:.0f}')

    # pairwise diversity over the 5x5 block grid
    grids = [_block_grid(lv.tiles) for lv in levels]
    total_pairs = len(levels) * (len(levels) - 1) // 2
    n_pairs = min(args.pairs, total_pairs)
    rng = random.Random(0)
    dists, seen = [], set()
    while len(dists) < n_pairs:
        a, b = rng.randrange(len(grids)), rng.randrange(len(grids))
        if a == b or (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        dists.append(_hamming(grids[a], grids[b]))
    if dists:
        print(f'  diversity (Hamming over 5x5 blocks, {len(dists)} pairs): '
              f'mean={statistics.mean(dists):.2f}')

    fieldnames = FIELDS + (['solver_steps'] if args.verify else [])
    out = Path(args.out) if args.out else Path(f'{args.preset}_report.csv')
    with out.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f'wrote {out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
