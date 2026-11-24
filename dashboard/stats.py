"""Statistics for the dashboard, standard library only.

Every proportion is reported with its sample size and a 95 % interval:
Wilson for one proportion, Newcombe (hybrid score, method 10) for a
difference of two, and a seeded bootstrap for the mean of a few runs.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> dict:
    """Wilson score interval for ``k`` successes out of ``n``."""
    if n <= 0:
        return {'k': k, 'n': n, 'p': None, 'lo': None, 'hi': None}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return {'k': k, 'n': n, 'p': p, 'lo': max(0.0, centre - half), 'hi': min(1.0, centre + half)}


def newcombe(k1: int, n1: int, k2: int, n2: int, z: float = Z95) -> dict:
    """Difference ``p1 - p2`` with Newcombe's hybrid score interval."""
    if n1 <= 0 or n2 <= 0:
        return {'d': None, 'lo': None, 'hi': None, 'n1': n1, 'n2': n2}
    a, b = wilson(k1, n1, z), wilson(k2, n2, z)
    p1, p2 = a['p'], b['p']
    d = p1 - p2
    lo = d - math.sqrt((p1 - a['lo']) ** 2 + (b['hi'] - p2) ** 2)
    hi = d + math.sqrt((a['hi'] - p1) ** 2 + (p2 - b['lo']) ** 2)
    return {'d': d, 'lo': lo, 'hi': hi, 'n1': n1, 'n2': n2}


def quantile(sorted_values: Sequence[float], q: float):
    """Linear-interpolation quantile (numpy's default) of sorted values."""
    n = len(sorted_values)
    if n == 0:
        return None
    pos = (n - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summary(values: Sequence[float]) -> dict:
    v = sorted(values)
    return {'n': len(v), 'median': quantile(v, 0.5), 'q1': quantile(v, 0.25),
            'q3': quantile(v, 0.75), 'min': v[0] if v else None, 'max': v[-1] if v else None,
            'mean': sum(v) / len(v) if v else None}


def lttb(xs: Sequence[float], ys: Sequence[float], threshold: int) -> list[int]:
    """Largest-Triangle-Three-Buckets downsampling: indices of the kept points
    (always the first and last; one point per bucket otherwise)."""
    n = len(xs)
    if threshold >= n or threshold < 3:
        return list(range(n)) if threshold >= n else ([0, n - 1] if n > 1 else list(range(n)))
    out = [0]
    every = (n - 2) / (threshold - 2)
    a = 0
    for i in range(threshold - 2):
        start = int(math.floor((i + 1) * every)) + 1
        end = min(int(math.floor((i + 2) * every)) + 1, n)
        avg_x = sum(xs[start:end]) / (end - start) if end > start else xs[n - 1]
        avg_y = sum(ys[start:end]) / (end - start) if end > start else ys[n - 1]
        r0 = int(math.floor(i * every)) + 1
        r1 = int(math.floor((i + 1) * every)) + 1
        ax, ay = xs[a], ys[a]
        best, best_area = r0, -1.0
        for j in range(r0, r1):
            area = abs((ax - avg_x) * (ys[j] - ay) - (ax - xs[j]) * (avg_y - ay))
            if area > best_area:
                best_area, best = area, j
        out.append(best)
        a = best
    out.append(n - 1)
    return out


def interp(xs: Sequence[float], ys: Sequence[float], grid: Sequence[float]) -> list:
    """Linear interpolation of (xs, ys) at ``grid`` (``None`` outside xs)."""
    out = []
    j = 0
    for g in grid:
        if not xs or g < xs[0] or g > xs[-1]:
            out.append(None)
            continue
        while j + 1 < len(xs) and xs[j + 1] < g:
            j += 1
        if j + 1 >= len(xs) or xs[j] == g:
            out.append(ys[j])
            continue
        x0, x1 = xs[j], xs[j + 1]
        t = (g - x0) / (x1 - x0) if x1 != x0 else 0.0
        out.append(ys[j] + (ys[j + 1] - ys[j]) * t)
    return out


def bootstrap_mean_ci(samples: Sequence[Sequence[float]], resamples: int = 1000,
                      seed: int = 0, level: float = 0.95) -> dict:
    """Per grid point: mean over runs and a percentile bootstrap interval.

    ``samples[r][i]`` is run ``r`` at grid point ``i`` (``None`` = missing).
    The RNG is seeded, so the band is the same on every request.
    """
    rng = random.Random(seed)
    npts = len(samples[0]) if samples else 0
    mean, lo, hi, n = [], [], [], []
    alpha = (1 - level) / 2
    for i in range(npts):
        vals = [s[i] for s in samples if s[i] is not None]
        n.append(len(vals))
        if not vals:
            mean.append(None), lo.append(None), hi.append(None)
            continue
        m = sum(vals) / len(vals)
        mean.append(m)
        if len(vals) < 2:
            lo.append(m), hi.append(m)
            continue
        k = len(vals)
        boots = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k
                       for _ in range(resamples))
        lo.append(quantile(boots, alpha))
        hi.append(quantile(boots, 1 - alpha))
    return {'mean': mean, 'lo': lo, 'hi': hi, 'n': n}


def ema(values: Sequence[float], alpha: float) -> list[float]:
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def buckets_by_quantile(values: Sequence[float], k: int = 5) -> list[float]:
    """Edges of ``k`` quantile buckets (deduplicated)."""
    v = sorted(values)
    if not v:
        return []
    edges = [quantile(v, i / k) for i in range(k + 1)]
    out = []
    for e in edges:
        if not out or e > out[-1]:
            out.append(e)
    return out
