"""Shared fixtures for the dashboard tests: a synthetic runs/ folder."""
import csv
import json
import time
from pathlib import Path

from game import Level
from game.replay import reference_replays, rle_encode, simulate
from game.replay_store import append_jsonl, append_replays, atomic_write_json, store_level

ROOT = Path(__file__).resolve().parents[1]
HEADER = ['update', 'env_steps', 'sps', 'episodes', 'return', 'success', 'gem_rate', 'pg_loss',
          'v_loss', 'entropy', 'approx_kl', 'eval_train', 'eval_test']


def write_progress(run: Path, rows: int, start: int = 1):
    new = not (run / 'progress.csv').exists()
    with open(run / 'progress.csv', 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(HEADER)
        for u in range(start, start + rows):
            s = min(1.0, u / max(1, rows))
            w.writerow([u, u * 4096, 3000, u * 10, 2 * s, s, s, -0.01, 0.01, 1.7 - s, 0.01,
                        s if u % 10 == 0 else '', s / 2 if u % 10 == 0 else ''])


def make_runs(root: Path, n_rows: int = 120) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    # 1. A legacy run: config/progress/eval only (no status, no replays).
    a = root / 'legacy'
    a.mkdir()
    (a / 'config.json').write_text(json.dumps({'preset': 'tiny', 'num_levels': 10, 'seed': 1,
                                               'start_seed': 0, 'levels': '1-6'}))
    write_progress(a, n_rows)
    (a / 'eval.json').write_text(json.dumps({
        'sampled': {'train': {'episodes': 64, 'success_rate': 1.0, 'gem_rate': 1.0, 'mean_return': 2.6},
                    'test': {'episodes': 64, 'success_rate': 0.3125, 'gem_rate': 0.4, 'mean_return': 0.9}},
        'greedy': {'train': {'episodes': 64, 'success_rate': 1.0, 'gem_rate': 1.0, 'mean_return': 2.6},
                   'test': {'episodes': 64, 'success_rate': 0.2, 'gem_rate': 0.3, 'mean_return': 0.6}},
        'train_seconds': 100.0}))
    (a / 'model.pt').write_bytes(b'not a real model')
    # 2. Three seeds of one group, with status/metrics/replays.
    refs = reference_replays(ROOT / 'levels' / 'solutions')
    for s in (1, 2, 3):
        r = root / f'grp_s{s}'
        r.mkdir()
        (r / 'config.json').write_text(json.dumps({'levels': '1-3', 'preset': None, 'seed': s,
                                                   'name': r.name, 'group': 'grp'}))
        write_progress(r, n_rows // (s + 1) + 5)
        atomic_write_json(r / 'status.json', {
            'schema': 1, 'name': r.name, 'group': 'grp', 'state': 'finished', 'pid': None,
            'started_at': '2026-01-01T00:00:00Z', 'updated_at': '2026-01-01T00:10:00Z',
            'update': 10, 'updates_total': 10, 'env_steps': 40960, 'total_steps': 40960})
    recs = []
    for ref in refs[:3]:
        level = Level.from_json(ROOT / 'levels' / f"level_{ref['n']:03d}.json")
        store_level(root, level)
        rec = {k: v for k, v in ref.items() if k not in ('n', 'id', 'run')}
        recs.append(dict(rec, policy='greedy', source={'kind': 'final', 'ckpt': 'model.pt'},
                         created='2026-01-01T00:00:00Z'))
    append_replays(root / 'grp_s1', recs)
    # A sweep on grp_s1.
    lines = []
    lv = Level.from_json(ROOT / 'levels' / 'level_003.json')
    digest = store_level(root, lv)
    for i in range(60):
        acts = rle_encode([5] * (30 + i % 7) + [2] * 40) if i % 3 else [[2, 200]]
        t = simulate(lv, acts, frame_skip=4, record=False)
        lines.append({'schema': 1, 'i': i, 'level_ref': {'kind': 'builtin', 'n': 3},
                      'level_hash': digest, 'policy': 'sampled', 'success': t.result == 'won',
                      'gem': t.gem_step is not None, 'engine_steps': t.engine_steps,
                      'env_steps': t.env_steps, 'end_reason': t.end_reason, 'gem_step': t.gem_step,
                      'win_step': t.win_step,
                      'death_xy': None if t.death_xy is None else list(t.death_xy),
                      'level_metrics': {'width': 240, 'height': 160, 'min_window': i % 4},
                      'action_hist': [0, 0, 1, 0, 0, 1], 'split': 'builtin', 'frame_skip': 4,
                      'actions': acts})
    append_jsonl(root / 'grp_s1' / 'sweeps' / 's1.jsonl', lines)
    atomic_write_json(root / 'grp_s1' / 'sweeps' / 's1.meta.json',
                      {'tag': 's1', 'n': len(lines), 'policy': 'sampled', 'ckpt': 'model.pt'})
    return root


def client(root, **kw):
    from fastapi.testclient import TestClient
    from dashboard.app import create_app
    app = create_app(root, watch=kw.pop('watch', False), find_python=False,
                     token=kw.pop('token', 'tok'), **kw)
    return TestClient(app, base_url='http://127.0.0.1:8765'), app


WRITE_HEADERS = {'Origin': 'http://127.0.0.1:8765', 'X-Dashboard-Token': 'tok'}


def wait_for(pred, timeout=5.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(step)
    return False
