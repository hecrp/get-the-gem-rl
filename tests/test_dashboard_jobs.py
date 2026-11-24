"""Jobs: whitelist vs the real CLIs, lifecycle, queue limits, adoption."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip('fastapi')

from dashboard import job_schema  # noqa: E402
from dashboard.jobs import JobManager  # noqa: E402
from dashboard_helpers import wait_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _help(module):
    return subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT, capture_output=True,
                          text=True, timeout=120).stdout


@pytest.mark.parametrize('kind,module', [('train', 'rl.ppo'), ('rollout', 'rl.rollout'),
                                         ('sweep', 'rl.rollout')])
def test_every_whitelisted_flag_exists_in_the_real_cli(kind, module):
    pytest.importorskip('torch')
    text = _help(module)
    for key in job_schema.SCHEMAS[kind]:
        flag = '--' + key.replace('_', '-')
        if key == 'policy':
            assert '--greedy' in text and '--sampled' in text
        else:
            assert flag in text, f'{flag} missing from {module} --help'


def _fake_python(tmp_path, body):
    """A stand-in interpreter: a script that ignores ``-m module`` and runs ``body``."""
    script = tmp_path / 'fakepy'
    script.write_text(f'#!{sys.executable}\nimport sys, time, json, os, signal\n{body}\n')
    script.chmod(0o755)
    return str(script)


def test_lifecycle_queue_limits_and_stop(tmp_path):
    runs = tmp_path / 'runs'
    for r in ('a', 'b'):
        (runs / r).mkdir(parents=True)
        (runs / r / 'model.pt').write_bytes(b'x')
    py = _fake_python(tmp_path, 'signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))\n'
                                'print("started", sys.argv[1:], flush=True)\ntime.sleep(30)')
    seen = []
    jm = JobManager(runs, py, on_change=lambda j: seen.append((j.id, j.state)))
    jm.start()
    try:
        j1 = jm.submit('rollout', {'run': 'a'})
        j2 = jm.submit('sweep', {'run': 'b', 'preset': 'tiny', 'num': 10})
        assert j1.state == 'running' and j2.state == 'queued'      # record limit = 1
        assert wait_for(lambda: 'started' in jm.tail(j1))
        assert '--mode' in j1.argv and j1.argv[1:3] == ['-m', 'rl.rollout']
        jm.stop(j2.id)                                               # cancel queued
        assert j2.state == 'stopped'
        with pytest.raises(job_schema.JobError) as e:
            jm.stop(j1.id, force=True)                               # too early
        assert e.value.status == 409
        jm.stop(j1.id)
        assert wait_for(lambda: j1.state == 'stopped', 10)
        assert j1.returncode == 0 and (1, 'running') in seen
    finally:
        jm.close()


def test_failed_job_records_error_and_continuations(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'r').mkdir(parents=True)
    (runs / 'r' / 'model.pt').write_bytes(b'x')
    py = _fake_python(tmp_path, 'print("boom: bad thing", flush=True)\nsys.exit(3)')
    jm = JobManager(runs, py)
    jm.start()
    try:
        j = jm.submit('rollout', {'run': 'r'})
        assert wait_for(lambda: j.state == 'failed', 10)
        assert j.returncode == 3 and 'boom' in j.error
    finally:
        jm.close()
    ok = _fake_python(tmp_path, 'sys.exit(0)')
    jm = JobManager(runs, ok)
    jm.start()
    try:
        j = jm.submit('rollout', {'run': 'r'}, then=[{'kind': 'sweep', 'args': {'num': 5}}])
        assert wait_for(lambda: len(jm.jobs) == 2 and all(x.state == 'finished' for x in jm.jobs.values()), 10)
        cont = jm.jobs[2]
        assert cont.kind == 'sweep' and cont.args['run'] == 'r'
    finally:
        jm.close()


def test_adopts_running_external_runs(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'ext').mkdir(parents=True)
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        (runs / 'ext' / 'status.json').write_text(json.dumps({'state': 'running', 'pid': proc.pid}))
        (runs / 'dead').mkdir()
        (runs / 'dead' / 'status.json').write_text(json.dumps({'state': 'running', 'pid': 999_999}))
        jm = JobManager(runs, sys.executable)
        jm.adopt()
        jobs = jm.list()
        assert len(jobs) == 1 and jobs[0]['run'] == 'ext' and jobs[0]['external']
        jm.stop(jobs[0]['id'])                                     # SIGTERM to the pid
        proc.wait(10)
        jm.poll()
        assert jm.get(jobs[0]['id']).state in ('stopped', 'finished', 'failed')
    finally:
        if proc.poll() is None:
            proc.kill()


def test_minimal_environment_and_cwd(tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'r').mkdir(parents=True)
    (runs / 'r' / 'model.pt').write_bytes(b'x')
    py = _fake_python(tmp_path, 'print(json.dumps({"cwd": os.getcwd(), "env": sorted(os.environ)}), flush=True)')
    os.environ['SECRET_TOKEN_FOR_TEST'] = 'x'
    jm = JobManager(runs, py)
    jm.start()
    try:
        j = jm.submit('rollout', {'run': 'r'})
        assert wait_for(lambda: j.state == 'finished', 10)
        out = json.loads(jm.tail(j).splitlines()[-1])
        assert out['cwd'] == str(ROOT)
        assert 'SECRET_TOKEN_FOR_TEST' not in out['env'] and 'PYTHONUNBUFFERED' in out['env']
    finally:
        jm.close()
        del os.environ['SECRET_TOKEN_FOR_TEST']
