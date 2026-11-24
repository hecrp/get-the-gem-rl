"""A real server: SSE events within 3 s (test 16) and demo mode without torch (test 18)."""
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip('fastapi')
httpx = pytest.importorskip('httpx')
uvicorn = pytest.importorskip('uvicorn')

from dashboard_helpers import make_runs, write_progress  # noqa: E402
from game.replay import reference_replays  # noqa: E402
from game.replay_store import append_replays  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path):
    from dashboard.app import create_app
    root = make_runs(tmp_path / 'runs', n_rows=20)
    app = create_app(root, find_python=False, token='tok', poll_s=0.2, heartbeat_s=1.0)
    port = _port()
    srv = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    end = time.time() + 10
    while not srv.started and time.time() < end:
        time.sleep(0.05)
    yield f'http://127.0.0.1:{port}', root
    srv.should_exit = True
    t.join(5)


def _collect(url, stop_after, timeout):
    events, end = [], time.time() + timeout
    with httpx.stream('GET', url + '/api/events', timeout=timeout) as r:
        name = None
        for line in r.iter_lines():
            if line.startswith('event: '):
                name = line[7:]
            elif line.startswith('data: '):
                events.append((name, json.loads(line[6:]), time.time()))
                if stop_after(events) or time.time() > end:
                    break
    return events


def test_sse_reports_progress_and_new_replays_quickly(server):
    url, root = server
    done = {}

    def writer():
        time.sleep(0.6)
        done['t0'] = time.time()
        write_progress(root / 'grp_s2', 3, start=500)
        rec = {k: v for k, v in reference_replays(ROOT / 'levels' / 'solutions')[0].items()
               if k not in ('n', 'id', 'run')}
        append_replays(root / 'grp_s2', [rec])

    threading.Thread(target=writer, daemon=True).start()

    def enough(evs):
        names = [e[0] for e in evs]
        return 'replay.added' in names and any(
            e[0] == 'run.updated' and 'progress' in e[1].get('changed', []) for e in evs)

    evs = _collect(url, enough, timeout=8)
    names = [e[0] for e in evs]
    assert names[0] == 'hello'
    added = [e for e in evs if e[0] == 'replay.added']
    upd = [e for e in evs if e[0] == 'run.updated' and 'progress' in e[1].get('changed', [])]
    assert added and added[0][1]['id'] == 'grp_s2/0001' and added[0][1]['run'] == 'grp_s2'
    assert upd and upd[0][1]['name'] == 'grp_s2'
    assert added[0][2] - done['t0'] < 3 and upd[0][2] - done['t0'] < 3


def test_sse_heartbeat(server):
    url, _ = server
    evs = _collect(url, lambda e: any(x[0] == 'heartbeat' for x in e), timeout=5)
    assert any(e[0] == 'heartbeat' for e in evs)


@pytest.mark.slow
def test_demo_mode_starts_without_torch(tmp_path):
    port = _port()
    code = ("import sys; sys.modules['torch'] = None; import runpy; "
            "sys.argv = ['dashboard', '--demo', '--no-jobs', '--port', '%d', '--runs-dir', %r]; "
            "runpy.run_module('dashboard', run_name='__main__')" % (port, str(tmp_path / 'runs')))
    proc = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        url = f'http://127.0.0.1:{port}'
        end = time.time() + 120
        while time.time() < end:
            try:
                h = httpx.get(url + '/api/health', timeout=1).json()
                break
            except httpx.HTTPError:
                time.sleep(0.3)
        else:
            pytest.fail('demo server did not start: ' + (proc.stdout.read() if proc.poll() is not None else ''))
        assert h['demo'] is True and h['torch'] is False
        runs = httpx.get(url + '/api/runs').json()['runs']
        assert len(runs) >= 5 and all(r['demo'] for r in runs)
        groups = httpx.get(url + '/api/groups?cols=success').json()['groups']
        assert any(len(g['runs']) == 3 and g['cols']['success']['band'] for g in groups)
        reps = httpx.get(url + '/api/replays?run=demo_policies&limit=5').json()
        assert reps['total'] > 40 and all(x['demo'] for x in reps['items'])
        t = httpx.get(url + f"/api/replays/{reps['items'][0]['id']}/trajectory").json()
        assert t['engine_steps'] > 0
        sw = httpx.get(url + '/api/sweeps/demo_policies/scripted_tiny/summary').json()
        assert sw['n'] == 400 and sw['demo'] is True
        assert (tmp_path / 'runs' / '_demo' / 'DEMO.json').exists()
    finally:
        proc.terminate()
        proc.wait(10)
