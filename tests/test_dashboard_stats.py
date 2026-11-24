"""dashboard.stats: LTTB, Wilson, Newcombe, bootstrap (spec 13 tests 13-14).

Reference intervals are literal values published in Newcombe (1998),
"Two-sided confidence intervals for the single proportion" and "Interval
estimation for the difference between independent proportions" (method 10).
"""
import math
import random

import pytest

from dashboard.stats import bootstrap_mean_ci, interp, lttb, newcombe, quantile, summary, wilson


@pytest.mark.parametrize('k,n,lo,hi', [
    (81, 263, 0.2553, 0.3662), (15, 148, 0.0624, 0.1605), (0, 20, 0.0, 0.1611),
    (1, 29, 0.0061, 0.1718), (29, 29, 0.8830, 1.0)])
def test_wilson_reference_values(k, n, lo, hi):
    w = wilson(k, n)
    assert round(w['lo'], 4) == lo and round(w['hi'], 4) == hi
    assert w['n'] == n and w['k'] == k


@pytest.mark.parametrize('a,b,lo,hi', [
    ((56, 70), (48, 80), 0.0524, 0.3339), ((9, 10), (3, 10), 0.1705, 0.8090),
    ((6, 7), (2, 7), 0.0582, 0.8062), ((5, 56), (0, 29), -0.0381, 0.1926)])
def test_newcombe_reference_values(a, b, lo, hi):
    d = newcombe(*a, *b)
    assert round(d['lo'], 4) == lo and round(d['hi'], 4) == hi


def test_wilson_and_newcombe_edge_cases():
    assert wilson(0, 0)['p'] is None
    assert newcombe(1, 0, 1, 2)['d'] is None


def test_lttb_keeps_ends_extremes_and_size():
    rng = random.Random(0)
    xs = list(range(10_000))
    ys = [math.sin(i / 300) + rng.gauss(0, 0.05) for i in xs]
    ys[4321] = 25.0            # a spike
    ys[7777] = -30.0           # a dip
    idx = lttb(xs, ys, 500)
    assert len(idx) == 500 and idx[0] == 0 and idx[-1] == 9999
    assert idx == sorted(set(idx))
    assert 4321 in idx and 7777 in idx
    kept = [ys[i] for i in idx]
    assert max(kept) == max(ys) and min(kept) == min(ys)
    assert lttb(xs[:10], ys[:10], 50) == list(range(10))


def test_quantiles_and_summary():
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    assert quantile([10], 0.9) == 10
    s = summary([5, 1, 3])
    assert (s['median'], s['q1'], s['q3'], s['n']) == (3, 2, 4, 3)


def test_interp():
    assert interp([0, 10], [0, 1], [0, 5, 10, 11]) == [0, 0.5, 1, None]


def test_bootstrap_band():
    runs = [[0.1, 0.5, 0.9], [0.2, 0.6, 1.0], [0.3, 0.7, 0.8]]
    a = bootstrap_mean_ci(runs, resamples=1000, seed=0)
    b = bootstrap_mean_ci(runs, resamples=1000, seed=0)
    assert a == b                                   # fixed seed: stable band
    for i, m in enumerate(a['mean']):
        assert a['lo'][i] <= m <= a['hi'][i]
        assert min(r[i] for r in runs) <= a['lo'][i] and a['hi'][i] <= max(r[i] for r in runs)
    assert a['mean'][0] == pytest.approx(0.2) and a['n'] == [3, 3, 3]
    const = bootstrap_mean_ci([[1.0], [1.0], [1.0]])
    assert const['lo'] == const['hi'] == [1.0]
    gaps = bootstrap_mean_ci([[1.0, None], [2.0, None], [3.0, 5.0]])
    assert gaps['n'] == [3, 1] and gaps['mean'][1] == 5.0
