"""Demo data without PyTorch (R-19): reference solutions, scripted/random
policies replayed with ``game.replay.simulate``, and synthetic training curves.

Everything written here is flagged ``demo: true`` and shown with a "DEMO DATA"
badge; nothing is presented as a real training result.
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
from pathlib import Path

from game import Level
from game.replay import reference_replays, rle_encode, simulate
from game.replay_store import append_jsonl, append_replays, atomic_write_json, store_level

from .store import BUILTIN_DIR, SOLUTIONS_DIR

DEMO_VERSION = 2
CREATED = '2026-01-01T00:00:00Z'          # fixed: demo data is reproducible


def _builtin(n):
    return Level.from_json(BUILTIN_DIR / f'level_{n:03d}.json')


def _tiny(seed):
    from pcg import PRESETS, generate_level, pcg_ref
    return generate_level(seed, PRESETS['tiny']), pcg_ref(seed, PRESETS['tiny'], 'tiny')


def _policy_actions(kind: str, rng: random.Random, max_steps: int, frame_skip: int):
    n = max_steps // frame_skip + 1
    if kind == 'right':
        return [[5, n]]
    return rle_encode(rng.randrange(6) for _ in range(n))


def _episode(level, ref, kind, rng, frame_skip=4):
    max_steps = level.max_steps or 1800
    actions = _policy_actions(kind, rng, max_steps, frame_skip)
    traj = simulate(level, actions, frame_skip=frame_skip)
    # Keep only the actions actually played.
    played, left = [], traj.env_steps
    for a, r in actions:
        if left <= 0:
            break
        played.append([a, min(r, left)])
        left -= r
    return played, traj, max_steps


def _replay(level, ref, digest, kind, actions, traj, max_steps, frame_skip, split, tags=()):
    policy = 'random' if kind == 'random' else f'scripted:{kind}'
    return {'schema': 1, 'created': CREATED,
            'source': {'kind': 'random' if kind == 'random' else 'scripted', 'demo': True},
            'policy': policy, 'split': split,
            'level': {'ref': ref, 'hash': digest, 'name': level.name, 'width': level.width,
                      'height': level.height},
            'env': {'frame_skip': frame_skip, 'max_steps': max_steps, 'physics': None},
            'actions': actions, 'outcome': traj.outcome(), 'tags': ['demo', *tags]}


def _sweep_line(i, level, ref, digest, kind, actions, traj, split, frame_skip):
    from pcg import metrics
    hist = [0] * 6
    for a, r in actions:
        hist[a] += r
    lm = {'width': level.width, 'height': level.height}
    if ref.get('kind') == 'pcg':
        m = metrics(level)
        lm.update(min_window=m.min_window, mean_window=m.mean_window,
                  linearity=round(m.linearity, 4), leniency=round(m.leniency, 4),
                  density=round(m.density, 4), path_ledges=m.path_ledges)
    return {'schema': 1, 'i': i, 'level_ref': ref, 'level_hash': digest,
            'policy': 'random' if kind == 'random' else f'scripted:{kind}',
            'success': traj.result == 'won', 'gem': traj.gem_step is not None,
            'engine_steps': traj.engine_steps, 'env_steps': traj.env_steps,
            'end_reason': traj.end_reason, 'gem_step': traj.gem_step,
            'win_step': traj.win_step,
            'death_xy': None if traj.death_xy is None else list(traj.death_xy),
            'level_metrics': lm, 'action_hist': hist, 'split': split,
            'frame_skip': frame_skip, 'actions': actions, 'demo': True}


def _logistic(x, mid, k):
    return 1 / (1 + math.exp(-k * (x - mid)))


def _synthetic_run(root: Path, name: str, seed: int, group: str, mid: float, test_cap: float,
                   steps=5_000_000, rows=48):
    rng = random.Random(seed * 7919)
    d = root / name
    d.mkdir(parents=True)
    config = {'preset': 'tiny', 'levels': '1-6', 'num_levels': 100 if 'n100' in name else 10,
              'start_seed': 0, 'total_steps': steps, 'distance_shaping': 1.0,
              'gem_reward': 0.5, 'frame_skip': 4, 'seed': seed, 'name': name, 'group': group,
              'num_envs': 32, 'num_steps': 128, 'demo': True}
    (d / 'config.json').write_text(json.dumps(config, indent=2))
    header = ['update', 'env_steps', 'sps', 'episodes', 'return', 'success', 'gem_rate',
              'pg_loss', 'v_loss', 'entropy', 'approx_kl', 'eval_train', 'eval_test']
    batch = 4096
    lines, eps = [], 0
    with open(d / 'progress.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(1, rows + 1):
            u = round(i * steps / batch / rows)
            x = u * batch / steps
            s = min(1.0, max(0.0, _logistic(x, mid, 12) + rng.gauss(0, 0.03)))
            eps += int(300 + 900 * s)
            ev_tr = ev_te = ''
            if i % 4 == 0:
                ev_tr = round(min(1.0, s + rng.gauss(0.02, 0.02)), 4)
                ev_te = round(max(0.0, min(test_cap, s * test_cap + rng.gauss(0, 0.04))), 4)
            row = [u, u * batch, 2900 + rng.randrange(-80, 80), eps, round(0.1 + 2.5 * s, 4),
                   round(s, 4), round(min(1, s + 0.15), 4), round(rng.gauss(-0.02, 0.01), 5),
                   round(abs(rng.gauss(0.01, 0.004)), 5),
                   round(1.75 - 1.1 * s + rng.gauss(0, 0.02), 4),
                   round(abs(rng.gauss(0.012, 0.004)), 5), ev_tr, ev_te]
            w.writerow(row)
            rec = dict(zip(header, row))
            rec.update(eval_train=ev_tr if ev_tr != '' else None,
                       eval_test=ev_te if ev_te != '' else None,
                       t=round(u * batch / 2900, 1), lr=5e-4 * (1 - x),
                       ep_len_mean=round(1600 - 1300 * s), time_left_frac=round(0.5 + 0.3 * s, 3),
                       action_hist=[rng.randrange(200, 900) for _ in range(6)])
            lines.append(rec)
    append_jsonl(d / 'metrics.jsonl', lines)
    final = min(1.0, _logistic(1.0, mid, 12))
    ev = {}
    for pol, f in (('sampled', 1.0), ('greedy', 0.9)):
        ev[pol] = {'train': {'episodes': 64, 'success_rate': round(final * f, 4),
                             'gem_rate': round(final * f, 4), 'mean_return': 2.5 * final},
                   'test': {'episodes': 64, 'success_rate': round(final * test_cap * f, 4),
                            'gem_rate': round(final * test_cap * f, 4),
                            'mean_return': 2.5 * final * test_cap}}
    ev['train_seconds'] = steps / 2900
    (d / 'eval.json').write_text(json.dumps(ev, indent=2))
    atomic_write_json(d / 'status.json', {
        'schema': 1, 'name': name, 'group': group, 'state': 'finished', 'pid': None,
        'started_at': CREATED, 'updated_at': CREATED, 'update': lines[-1]['update'],
        'updates_total': lines[-1]['update'], 'env_steps': lines[-1]['env_steps'],
        'total_steps': steps, 'sps': 2900, 'device': 'demo', 'git_commit': None,
        'error': None, 'resumed_from': None, 'demo': True})
    return d


def build(root, force: bool = False) -> Path:
    """Create the demo runs folder (idempotent; rebuilt if outdated)."""
    root = Path(root)
    marker = root / 'DEMO.json'
    try:
        if not force and json.loads(marker.read_text()).get('version') == DEMO_VERSION:
            return root
    except (OSError, ValueError):
        pass
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rng = random.Random(2026)

    # 1. Scripted and random policies on built-in and generated levels.
    ref_run = root / 'demo_policies'
    ref_run.mkdir()
    (ref_run / 'config.json').write_text(json.dumps(
        {'levels': '1-9', 'preset': None, 'frame_skip': 4, 'seed': 0, 'name': 'demo_policies',
         'demo': True, 'note': 'scripted (hold RIGHT_JUMP) and uniform random policies'},
        indent=2))
    replays, lines = [], []
    builtin = [(n, _builtin(n), {'kind': 'builtin', 'n': n}) for n in range(1, 10)]
    for n, level, ref in builtin:
        digest = store_level(root, level)
        for kind in ('right', 'random', 'random'):
            acts, traj, ms = _episode(level, ref, kind, rng)
            replays.append(_replay(level, ref, digest, kind, acts, traj, ms, 4, 'builtin'))
    for seed in range(1_000_000, 1_000_012):
        level, ref = _tiny(seed)
        digest = store_level(root, level)
        for kind in ('right', 'random'):
            acts, traj, ms = _episode(level, ref, kind, rng)
            replays.append(_replay(level, ref, digest, kind, acts, traj, ms, 4, 'test'))
    # Reference solutions as stored replays too (they are also the virtual
    # ``_reference`` run): these are the true solutions, not demo results.
    for rec in reference_replays(SOLUTIONS_DIR, BUILTIN_DIR):
        store_level(root, _builtin(rec['n']))
        replays.append({k: v for k, v in rec.items() if k not in ('n', 'id', 'run')}
                       | {'created': CREATED, 'tags': ['demo', 'reference']})
    append_replays(ref_run, replays)

    # 2. Sweeps (random policy): built-in levels and 200 generated levels.
    sweeps = ref_run / 'sweeps'
    i = 0
    for n, level, ref in builtin:
        digest = store_level(root, level)
        for _ in range(25):
            acts, traj, _ = _episode(level, ref, 'random', rng)
            lines.append(_sweep_line(i, level, ref, digest, 'random', acts, traj, 'builtin', 4))
            i += 1
    append_jsonl(sweeps / 'random_builtin.jsonl', lines)
    atomic_write_json(sweeps / 'random_builtin.meta.json', {
        'schema': 1, 'tag': 'random_builtin', 'policy': 'random', 'n': len(lines),
        'created': CREATED, 'levels': list(range(1, 10)), 'demo': True, 'ckpt': None})
    lines = []
    for k, seed in enumerate(range(1_000_000, 1_000_200)):
        level, ref = _tiny(seed)
        digest = store_level(root, level)
        for kind in ('right', 'random'):
            acts, traj, _ = _episode(level, ref, kind, rng)
            lines.append(_sweep_line(len(lines), level, ref, digest, kind, acts, traj, 'test', 4))
    append_jsonl(sweeps / 'scripted_tiny.jsonl', lines)
    atomic_write_json(sweeps / 'scripted_tiny.meta.json', {
        'schema': 1, 'tag': 'scripted_tiny', 'policy': 'scripted:right + random', 'n': len(lines),
        'created': CREATED, 'seed_range': [1_000_000, 1_000_199], 'demo': True, 'ckpt': None})
    atomic_write_json(ref_run / 'status.json', {
        'schema': 1, 'name': 'demo_policies', 'state': 'finished', 'pid': None,
        'started_at': CREATED, 'updated_at': CREATED, 'demo': True})

    # 3. Synthetic training curves: a 3-seed group (bands) and a memorising run.
    for s in (1, 2, 3):
        _synthetic_run(root, f'demo_tiny_n100_s{s}', s, 'demo_tiny_n100', 0.18 + 0.03 * s, 0.97)
    _synthetic_run(root, 'demo_tiny_n10_s1', 1, 'demo_tiny_n10', 0.15, 0.3)
    atomic_write_json(marker, {'version': DEMO_VERSION, 'demo': True,
                               'note': 'synthetic data generated by dashboard.demo'})
    return root
