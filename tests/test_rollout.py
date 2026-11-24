"""rl.rollout: replays, numbering under concurrency, level store, verify, sweeps."""
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

from game import Level                                   # noqa: E402
from game.level import level_hash                        # noqa: E402
from game.replay import simulate_replay                  # noqa: E402
from game.replay_store import append_replays, load_stored_level, read_jsonl  # noqa: E402
from rl.rollout import (builtin_ref, build_agent, load_policy, preset_refs,  # noqa: E402
                        record_replays, resolve_ref, sweep)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'levels': '1-3', 'preset': None, 'frame_skip': 4, 'gem_reward': 0.5,
          'distance_shaping': 1.0, 'no_goal_vector': False, 'start_seed': 0,
          'num_levels': 0}


def _make_run(runs, name='r', config=CONFIG):
    run = runs / name
    run.mkdir(parents=True)
    torch.manual_seed(0)
    agent = build_agent()
    torch.save({'model': agent.state_dict(), 'config': config}, run / 'model.pt')
    (run / 'config.json').write_text(json.dumps(config))
    return run


def test_record_replays_numbered_verified_and_stored(tmp_path):
    run = _make_run(tmp_path)
    policy = load_policy(run / 'model.pt')
    refs = [builtin_ref(n) for n in (1, 2, 3)]
    a = record_replays(policy, refs, run, tmp_path, greedy=True, n_envs=2)
    b = record_replays(policy, refs[:2], run, tmp_path, greedy=False, seed=3)
    assert [r['n'] for r in a + b] == [1, 2, 3, 4, 5]
    assert [r['id'] for r in a] == ['r/0001', 'r/0002', 'r/0003']
    recs, _ = read_jsonl(run / 'replays.jsonl')
    assert [r['n'] for r in recs] == [1, 2, 3, 4, 5]
    for r in recs:
        assert r['schema'] == 1 and r['run'] == 'r'
        assert r['source']['kind'] == 'final' and r['split'] == 'builtin'
        assert r['policy'] in ('greedy', 'sampled')
        level = load_stored_level(tmp_path, r['level']['hash'])
        assert level is not None and level_hash(level) == r['level']['hash']
        assert level == resolve_ref(r['level']['ref'])
        traj = simulate_replay(r, level)
        assert traj.result == r['outcome']['result']
        assert traj.engine_steps == r['outcome']['engine_steps']
        assert 'return' in r['outcome']


def test_numbers_never_reused_after_deletion(tmp_path):
    run = tmp_path / 'x'
    run.mkdir()
    append_replays(run, [{'k': 1}, {'k': 2}])
    (run / 'replays.jsonl').unlink()                 # user deletes every replay
    out = append_replays(run, [{'k': 3}])
    assert out[0]['n'] == 3 and out[0]['id'] == 'x/0003'


def _writer(run_dir, k):
    for _ in range(25):
        append_replays(run_dir, [{'writer': k}, {'writer': k}])


def test_concurrent_writers_get_distinct_monotonic_numbers(tmp_path):
    run = tmp_path / 'c'
    run.mkdir()
    ctx = mp.get_context('spawn')
    procs = [ctx.Process(target=_writer, args=(run, k)) for k in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    recs, _ = read_jsonl(run / 'replays.jsonl')
    ns = [r['n'] for r in recs]
    assert len(ns) == 150 and sorted(ns) == list(range(1, 151))
    assert ns == sorted(ns)                          # appended in number order


def test_cli_two_concurrent_rollouts(tmp_path):
    _make_run(tmp_path)
    cmd = [sys.executable, '-m', 'rl.rollout', '--runs-dir', str(tmp_path), '--run', 'r',
           '--levels', '1-3', '--threads', '1']
    procs = [subprocess.Popen(cmd + ['--seed', str(k), '--sampled'], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for k in range(2)]
    outs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), outs
    summaries = [json.loads(o[0].strip().splitlines()[-1]) for o in outs]
    ids = sorted(i for s in summaries for i in s['ids'])
    assert ids == [f'r/{n:04d}' for n in range(1, 7)]


def test_cli_verify_detects_corruption(tmp_path):
    _make_run(tmp_path)
    cmd = [sys.executable, '-m', 'rl.rollout', '--runs-dir', str(tmp_path), '--run', 'r',
           '--levels', '1', '--threads', '1', '--debug-corrupt']
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode != 0 and 'mismatch' in p.stderr
    assert not (tmp_path / 'r' / 'replays.jsonl').exists()


def test_cli_errors_are_clean(tmp_path):
    p = subprocess.run([sys.executable, '-m', 'rl.rollout', '--runs-dir', str(tmp_path),
                        '--run', 'missing'], cwd=ROOT, capture_output=True, text=True,
                       timeout=120)
    assert p.returncode != 0 and 'run not found' in p.stderr


def test_load_policy_mismatch_is_clear(tmp_path):
    bad = tmp_path / 'bad.pt'
    torch.save({'model': {'nope': torch.zeros(1)}, 'config': {}}, bad)
    with pytest.raises(ValueError, match='observation layout'):
        load_policy(bad)
    torch.save({'weights': 1}, bad)
    with pytest.raises(ValueError, match='not a GemEnv PPO checkpoint'):
        load_policy(bad)


def test_sweep_lines_and_meta(tmp_path):
    config = dict(CONFIG, preset='tiny', num_levels=10, levels='1-6')
    run = _make_run(tmp_path, 's', config)
    policy = load_policy(run / 'model.pt')
    refs = preset_refs('tiny', 1_000_000, 6) + preset_refs('tiny', 3, 2)
    meta = sweep(policy, refs, run, tmp_path, greedy=True, tag='t', keep=1, n_envs=4)
    lines, _ = read_jsonl(run / 'sweeps' / 't.jsonl')
    assert meta['n'] == 8 and len(lines) == 8
    assert [x['i'] for x in lines] == list(range(8))
    splits = [x['split'] for x in lines]
    assert splits[:6] == ['test'] * 6 and splits[6:] == ['train'] * 2
    for x in lines:
        assert x['level_ref']['kind'] == 'pcg'
        assert {'min_window', 'leniency', 'width', 'height'} <= set(x['level_metrics'])
        assert sum(x['action_hist']) == x['env_steps']
        assert x['end_reason'] in ('won', 'hazard', 'fall', 'timeout')
    meta_file = json.loads((run / 'sweeps' / 't.meta.json').read_text())
    assert meta_file['seed_range'] == [3, 1_000_005] and meta_file['policy'] == 'greedy'
    assert 1 <= len(meta_file['replays']) <= 3
    recs, _ = read_jsonl(run / 'replays.jsonl')
    assert all('sweep:t' in r['tags'] for r in recs)
