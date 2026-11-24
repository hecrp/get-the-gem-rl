"""rl.ppo observability: status.json, metrics.jsonl, checkpoints, replays,
--resume, SIGTERM and bit-exact behaviour without flags (spec 13 test 10)."""
import csv
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / 'fixtures'
BASE = ['--preset', 'tiny', '--num-levels', '10', '--num-envs', '4', '--num-steps', '128',
        '--device', 'cpu', '--threads', '1', '--eval-episodes', '4', '--eval-envs', '2',
        '--eval-levels', '10']


def _ppo(runs, *extra, timeout=300):
    cmd = [sys.executable, '-m', 'rl.ppo', *BASE, '--runs-dir', str(runs), *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def _rows(path):
    with open(path, newline='') as fh:
        return list(csv.reader(fh))


def test_status_and_metrics_always_written(tmp_path):
    p = _ppo(tmp_path, '--total-steps', '1024', '--log-every', '1', '--name', 'm')
    assert p.returncode == 0, p.stderr
    run = tmp_path / 'm'
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'finished' and status['update'] == 2
    assert status['updates_total'] == 2 and status['env_steps'] == 1024
    assert status['pid'] > 0 and status['error'] is None and status['schema'] == 1
    lines = [json.loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines()]
    header = _rows(run / 'progress.csv')[0]
    assert len(lines) == 2
    for rec in lines:
        assert set(header) <= set(rec)
        assert len(rec['action_hist']) == 6 and sum(rec['action_hist']) == 512
        assert 0 <= rec['time_left_frac'] <= 1 and rec['t'] > 0 and rec['lr'] > 0
    config = json.loads((run / 'config.json').read_text())
    assert config['group'] is None and config['ckpt_every'] == 0


@pytest.mark.slow
def test_no_flags_progress_identical_to_pre_phase5_run(tmp_path):
    meta = json.loads((FIXTURES / 'ppo_golden_meta.json').read_text())
    here = {'torch': torch.__version__, 'machine': platform.machine(),
            'system': platform.system(), 'python': platform.python_version()}
    if here != meta:
        pytest.skip(f'golden recorded on {meta}, this is {here}')
    p = _ppo(tmp_path, '--total-steps', '4096', '--log-every', '2', '--eval-every', '4',
             '--distance-shaping', '1.0', '--name', 'g')
    assert p.returncode == 0, p.stderr
    gold = _rows(FIXTURES / 'ppo_golden_progress.csv')
    new = _rows(tmp_path / 'g' / 'progress.csv')
    sps = gold[0].index('sps')                      # wall-clock dependent
    strip = lambda rows: [r[:sps] + r[sps + 1:] for r in rows]   # noqa: E731
    assert strip(new) == strip(gold)


@pytest.mark.slow
def test_checkpoints_replays_and_resume(tmp_path):
    p = _ppo(tmp_path, '--total-steps', '2048', '--log-every', '1', '--name', 'f',
             '--ckpt-every', '2', '--ckpt-keep', '1', '--replay-every', '2', '--replay-n', '2',
             '--group', 'exp')
    assert p.returncode == 0, p.stderr
    run = tmp_path / 'f'
    assert sorted(x.name for x in (run / 'ckpt').iterdir()) == ['update_000004.pt']
    ck = torch.load(run / 'ckpt' / 'update_000004.pt', weights_only=False)
    assert {'model', 'optimizer', 'update', 'env_steps', 'rng', 'config'} <= set(ck)
    assert ck['update'] == 4 and ck['env_steps'] == 2048
    reps = [json.loads(x) for x in (run / 'replays.jsonl').read_text().splitlines()]
    assert [r['n'] for r in reps] == list(range(1, len(reps) + 1)) and reps
    assert {r['source']['update'] for r in reps} <= {2, 4}
    assert all(r['source']['kind'] == 'checkpoint' for r in reps)
    assert {r['policy'] for r in reps} <= {'greedy', 'sampled'}
    assert json.loads((run / 'status.json').read_text())['group'] == 'exp'
    # Resume, extending the run: counters continue and files are appended.
    p = _ppo(tmp_path, '--name', 'f', '--resume', '--total-steps', '3072', '--log-every', '1')
    assert p.returncode == 0, p.stderr
    rows = _rows(run / 'progress.csv')
    assert rows[0][0] == 'update' and [r[0] for r in rows[1:]] == [str(u) for u in range(1, 7)]
    assert [int(r[1]) for r in rows[1:]] == [512 * u for u in range(1, 7)]
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'finished' and status['resumed_from'] == 'ckpt/update_000004.pt'
    assert status['env_steps'] == 3072
    metrics = (run / 'metrics.jsonl').read_text().splitlines()
    assert [json.loads(x)['update'] for x in metrics] == list(range(1, 7))


def test_sigterm_stops_cleanly(tmp_path):
    cmd = [sys.executable, '-m', 'rl.ppo', *BASE, '--runs-dir', str(tmp_path), '--name', 's',
           '--total-steps', '200000', '--log-every', '1']
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    status = tmp_path / 's' / 'status.json'
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if json.loads(status.read_text())['update'] >= 1:
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, err
    st = json.loads(status.read_text())
    assert st['state'] == 'stopped' and st['last_ckpt']
    ev = json.loads((tmp_path / 's' / 'eval.json').read_text())
    assert ev['partial'] is True and ev['stopped_at_update'] == st['update']
    from rl.rollout import load_policy
    load_policy(tmp_path / 's' / 'model.pt')
    assert (tmp_path / 's' / st['last_ckpt']).exists()


def test_failure_marks_status_failed(tmp_path):
    p = _ppo(tmp_path, '--total-steps', '1024', '--name', 'bad', '--minibatches', '0')
    assert p.returncode != 0
    st = json.loads((tmp_path / 'bad' / 'status.json').read_text())
    assert st['state'] == 'failed' and 'ZeroDivisionError' in st['error']
