"""Aggregates for sweeps and experiment groups (spec 13 §96), computed server-side."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from game.constants import PLAYER_H, PLAYER_W, TILE_SIZE
from game.replay import simulate

from .stats import (bootstrap_mean_ci, buckets_by_quantile, interp, newcombe, summary,
                    wilson)

END_REASONS = ('won', 'hazard', 'fall', 'timeout')
MIN_BUCKET = 20


def histogram(values, bins: int = 24) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {'n': 0, 'edges': [], 'counts': []}
    lo, hi = min(values), max(values)
    if hi == lo:
        return {'n': len(values), 'edges': [lo, hi + 1], 'counts': [len(values)]}
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / width))] += 1
    return {'n': len(values), 'edges': [lo + i * width for i in range(bins + 1)],
            'counts': counts}


def _bucketed(lines, key, edges=None, integer=False) -> list[dict]:
    pairs = [(x['level_metrics'].get(key) if key != 'area' else
              x['level_metrics'].get('width', 0) * x['level_metrics'].get('height', 0),
              x['success']) for x in lines if x.get('level_metrics')]
    pairs = [(v, s) for v, s in pairs if v is not None]
    if not pairs:
        return []
    out = []
    if integer:
        groups = defaultdict(list)
        for v, s in pairs:
            groups[int(math.floor(v))].append(s)
        for b in sorted(groups):
            k, n = sum(groups[b]), len(groups[b])
            out.append({'label': str(b), 'x0': b, 'x1': b + 1, **wilson(k, n),
                        'insufficient': n < MIN_BUCKET})
        return out
    edges = edges or buckets_by_quantile([v for v, _ in pairs], 5)
    if len(edges) < 2:
        k, n = sum(s for _, s in pairs), len(pairs)
        return [{'label': f'{edges[0]:g}' if edges else '-', 'x0': edges[0] if edges else None,
                 'x1': edges[0] if edges else None, **wilson(k, n),
                 'insufficient': n < MIN_BUCKET}]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        sel = [s for v, s in pairs if lo <= v < hi or (last and v == hi)]
        k, n = sum(sel), len(sel)
        out.append({'label': f'{lo:.3g}–{hi:.3g}', 'x0': lo, 'x1': hi, **wilson(k, n),
                    'insufficient': n < MIN_BUCKET})
    return out


def sweep_summary(lines: list[dict]) -> dict:
    lines = [x for x in lines if '_broken' not in x]
    n = len(lines)
    wins = sum(1 for x in lines if x.get('success'))
    gems = sum(1 for x in lines if x.get('gem'))
    reasons = Counter(x.get('end_reason') for x in lines)
    end = {r: wilson(reasons.get(r, 0), n) for r in END_REASONS}
    other = n - sum(reasons.get(r, 0) for r in END_REASONS)
    if other:
        end['incomplete'] = wilson(other, n)
    by_split = {}
    for split in sorted({x.get('split', 'other') for x in lines}):
        sel = [x for x in lines if x.get('split', 'other') == split]
        by_split[split] = wilson(sum(1 for x in sel if x.get('success')), len(sel))
    gap = None
    if 'train' in by_split and 'test' in by_split:
        a, b = by_split['train'], by_split['test']
        gap = newcombe(a['k'], a['n'], b['k'], b['n'])
    t_gem = [x['gem_step'] for x in lines if x.get('gem_step') is not None]
    g2e = [x['win_step'] - x['gem_step'] for x in lines
           if x.get('success') and x.get('win_step') is not None
           and x.get('gem_step') is not None and x['win_step'] != x['gem_step']]
    len_won = [x['engine_steps'] for x in lines if x.get('success')]
    len_other = [x['engine_steps'] for x in lines if not x.get('success')]
    actions = [0] * 6
    for x in lines:
        for i, c in enumerate(x.get('action_hist') or ()):
            if i < 6:
                actions[i] += c
    levels = Counter(x.get('level_hash') for x in lines)
    level_list = []
    for h, c in levels.most_common(50):
        x0 = next(x for x in lines if x.get('level_hash') == h)
        ref = x0.get('level_ref') or {}
        level_list.append({'hash': h, 'episodes': c, 'ref': ref,
                           'label': (f"builtin {ref.get('n')}" if ref.get('kind') == 'builtin'
                                     else f"{ref.get('preset')} · seed {ref.get('seed')}"
                                     if ref.get('kind') == 'pcg' else h[:15]),
                           'success': wilson(sum(1 for x in lines if x.get('level_hash') == h
                                                 and x.get('success')), c)})
    builtin = defaultdict(list)
    for x in lines:
        ref = x.get('level_ref') or {}
        if ref.get('kind') == 'builtin':
            builtin[ref['n']].append(bool(x.get('success')))
    return {
        'n': n, 'success': wilson(wins, n), 'gem': wilson(gems, n), 'end_reasons': end,
        'by_split': by_split, 'gen_gap': gap,
        'time_to_gem': summary(t_gem), 'gem_to_exit': summary(g2e),
        'ep_len': {'won': summary(len_won), 'other': summary(len_other)},
        'histograms': {'ep_len': histogram([x['engine_steps'] for x in lines]),
                       'ep_len_won': histogram(len_won), 'ep_len_other': histogram(len_other),
                       'time_to_gem': histogram(t_gem), 'gem_to_exit': histogram(g2e)},
        'action_hist': actions,
        'difficulty': {'min_window': _bucketed(lines, 'min_window', integer=True),
                       'area': _bucketed(lines, 'area'),
                       'leniency': _bucketed(lines, 'leniency')},
        'by_builtin': {str(k): wilson(sum(v), len(v)) for k, v in sorted(builtin.items())},
        'levels': level_list, 'min_bucket': MIN_BUCKET,
    }


def heatmap(lines: list[dict], level, digest: str, layer: str = 'visits',
            max_episodes: int = 400) -> dict:
    """Per-8x8-cell counts on one level: engine steps of the hitbox centre per
    episode (``visits``, re-simulated from the stored actions) or deaths."""
    cols, rows = level.width // TILE_SIZE, level.height // TILE_SIZE
    eps = [x for x in lines if x.get('level_hash') == digest]
    out = {'cols': cols, 'rows': rows, 'cell': TILE_SIZE, 'episodes': len(eps),
           'layer': layer, 'level_hash': digest}
    if layer == 'deaths':
        grids = {c: [0] * (cols * rows) for c in ('hazard', 'fall')}
        for x in eps:
            if x.get('death_xy') and x.get('end_reason') in grids:
                px, py = x['death_xy']
                c = min(cols - 1, max(0, int((px + PLAYER_W / 2) // TILE_SIZE)))
                r = min(rows - 1, max(0, int((py + PLAYER_H / 2) // TILE_SIZE)))
                grids[x['end_reason']][r * cols + c] += 1
        out['grids'] = grids
        out['max'] = max([max(g) for g in grids.values()] + [0])
        return out
    grid = [0.0] * (cols * rows)
    used = 0
    for x in eps:
        if not x.get('actions'):
            continue
        if used >= max_episodes:
            break
        used += 1
        traj = simulate(level, x['actions'], frame_skip=x.get('frame_skip', 4), record=True)
        xy = traj.xy
        for i in range(0, len(xy), 2):
            c = int((xy[i] + PLAYER_W / 2) // TILE_SIZE)
            r = int((xy[i + 1] + PLAYER_H / 2) // TILE_SIZE)
            if 0 <= c < cols and 0 <= r < rows:
                grid[r * cols + c] += 1
    if used:
        grid = [v / used for v in grid]
    out.update(grid=grid, used=used, max=max(grid) if grid else 0)
    return out


def group_bands(series_by_run: dict[str, tuple[list, list]], points: int = 120,
                seed: int = 0) -> dict:
    """Align runs on a common ``env_steps`` grid (their overlapping range) and
    return mean + 95 % bootstrap band; individual lines with < 3 runs."""
    runs = {k: v for k, v in series_by_run.items() if v[0]}
    if not runs:
        return {'runs': 0}
    lo = max(xs[0] for xs, _ in runs.values())
    hi = min(xs[-1] for xs, _ in runs.values())
    if hi <= lo:
        return {'runs': len(runs), 'band': None, 'warning': 'runs do not overlap'}
    grid = [lo + (hi - lo) * i / (points - 1) for i in range(points)]
    aligned = [interp(xs, ys, grid) for xs, ys in runs.values()]
    out = {'runs': len(runs), 'x': grid}
    if len(runs) >= 3:
        out['band'] = bootstrap_mean_ci(aligned, resamples=1000, seed=seed)
    else:
        out['band'] = None
        out['warning'] = f'{len(runs)} run(s): fewer than 3 seeds, no band'
    return out
