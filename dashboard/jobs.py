"""Training/recording jobs as subprocesses, with a queue and orphan adoption.

The server never imports torch (R-15, D-17): ``rl.ppo`` and ``rl.rollout`` run
in child processes launched with an argument list (no shell), a fixed working
directory and a minimal environment. The real state of a training run is its
``status.json``; a job adds the pid, the exit code and the log.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from game.replay_store import read_json

from . import job_schema
from .store import ROOT, pid_alive

TRAIN_LIMIT, RECORD_LIMIT = 2, 1
FORCE_AFTER_S = 30.0


def find_train_python(explicit: str | None = None) -> tuple[str | None, str]:
    """Interpreter for jobs: ``explicit``, else this one if it has torch, else
    ``.venv/bin/python``. Checked in a subprocess: torch is never imported here."""
    candidates = [explicit] if explicit else [sys.executable, str(ROOT / '.venv' / 'bin' / 'python')]
    for py in candidates:
        if not py or not Path(py).exists():
            continue
        try:
            ok = subprocess.run(
                [py, '-c', 'import importlib.util, sys; '
                 'sys.exit(0 if importlib.util.find_spec("torch") and '
                 'importlib.util.find_spec("gymnasium") else 1)'],
                capture_output=True, timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if ok:
            return py, 'ok'
    return None, ('PyTorch/Gymnasium not found for ' + (explicit or 'this interpreter or .venv')
                  + ': install requirements-train.txt or pass --train-python')


class Job:
    def __init__(self, jid: int, kind: str, args: dict, run: str, then=None, external=False):
        self.id, self.kind, self.args, self.run = jid, kind, args, run
        self.then = then or []
        self.external = external
        self.state = 'running' if external else 'queued'
        self.pid = None
        self.proc: subprocess.Popen | None = None
        self.returncode = None
        self.created = time.time()
        self.started = self.ended = None
        self.term_sent = None
        self.argv: list[str] = []
        self.log_path: Path | None = None
        self.error = None

    def public(self) -> dict:
        return {'id': self.id, 'kind': self.kind, 'args': self.args, 'run': self.run,
                'state': self.state, 'pid': self.pid, 'returncode': self.returncode,
                'created': self.created, 'started': self.started, 'ended': self.ended,
                'external': self.external, 'argv': self.argv[1:] if self.argv else [],
                'then': self.then, 'error': self.error,
                'log': str(self.log_path.relative_to(ROOT)) if self.log_path and
                self.log_path.is_relative_to(ROOT) else (str(self.log_path) if self.log_path else None),
                'term_sent': self.term_sent}


class JobManager:
    def __init__(self, runs_dir: Path, python: str | None, on_change=None,
                 train_limit=TRAIN_LIMIT, record_limit=RECORD_LIMIT):
        self.runs_dir = Path(runs_dir)
        self.python = python
        self.jobs: dict[int, Job] = {}
        self.queue: deque[int] = deque()
        self.lock = threading.RLock()
        self.on_change = on_change or (lambda job: None)
        self.limits = {'train': train_limit, 'record': record_limit}
        self._next = 1
        self._stop = threading.Event()
        self._thread = None

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        self.adopt()
        self._thread = threading.Thread(target=self._loop, name='jobs', daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()

    def adopt(self):
        """Adopt runs whose ``status.json`` says running with a live pid."""
        try:
            dirs = [p for p in self.runs_dir.iterdir() if p.is_dir()]
        except OSError:
            return
        for d in dirs:
            st = read_json(d / 'status.json')
            if not st or st.get('state') != 'running' or not pid_alive(st.get('pid')):
                continue
            with self.lock:
                if any(j.pid == st['pid'] for j in self.jobs.values()):
                    continue
                job = Job(self._next, 'train', {'name': d.name}, d.name, external=True)
                self._next += 1
                job.pid = st['pid']
                job.started = time.time()
                job.log_path = d / 'job.log' if (d / 'job.log').exists() else None
                self.jobs[job.id] = job
            self.on_change(job)

    # -- submission -------------------------------------------------------------

    def submit(self, kind: str, args, then=None) -> Job:
        if self.python is None:
            raise job_schema.JobError('training/recording disabled: no PyTorch interpreter',
                                      503, 'no_torch')
        args = job_schema.validate(kind, args)
        run = job_schema.run_name(kind, args)
        if not job_schema.valid_run_name(run):
            raise job_schema.JobError(f'invalid run name {run!r}')
        run_dir = self.runs_dir / run
        conts = []
        for c in then or []:
            if not isinstance(c, dict) or c.get('kind') not in ('sweep', 'rollout'):
                raise job_schema.JobError('then: expected [{"kind": "sweep"|"rollout", "args": {...}}]')
            cargs = dict(c.get('args') or {})
            cargs.setdefault('run', run)
            conts.append({'kind': c['kind'], 'args': job_schema.validate(c['kind'], cargs)})
        if kind == 'train':
            if run_dir.exists() and 'resume' not in args:
                i = 2
                while (self.runs_dir / f'{run}_{i}').exists():
                    i += 1
                raise job_schema.JobError(
                    f'run {run!r} already exists; use resume or another name (e.g. {run}_{i})',
                    409, 'run_exists')
            if 'resume' in args and not run_dir.exists():
                raise job_schema.JobError(f'run {run!r} not found', 404, 'not_found')
            with self.lock:
                for j in self.jobs.values():
                    if j.run == run and j.kind == 'train' and j.state in ('queued', 'running'):
                        raise job_schema.JobError(f'run {run!r} already has an active job',
                                                  409, 'busy')
        else:
            if not run_dir.is_dir():
                raise job_schema.JobError(f'run {run!r} not found', 404, 'not_found')
            ckpt = args.get('ckpt', 'model.pt')
            if not (run_dir / ckpt).exists():
                raise job_schema.JobError(f'{run}/{ckpt} not found', 404, 'not_found')
        with self.lock:
            job = Job(self._next, kind, args, run, then=conts)
            self._next += 1
            job.argv = job_schema.argv(kind, args, self.python, str(self.runs_dir))
            self.jobs[job.id] = job
            self.queue.append(job.id)
        self.on_change(job)
        self._schedule()
        return job

    def _active(self, cls: str) -> int:
        return sum(1 for j in self.jobs.values() if j.state == 'running' and
                   ('train' if j.kind == 'train' else 'record') == cls)

    def _schedule(self):
        started = []
        with self.lock:
            for jid in list(self.queue):
                job = self.jobs[jid]
                cls = 'train' if job.kind == 'train' else 'record'
                if self._active(cls) >= self.limits[cls]:
                    continue
                self.queue.remove(jid)
                self._launch(job)
                started.append(job)
        for job in started:
            self.on_change(job)

    def _launch(self, job: Job):
        run_dir = self.runs_dir / job.run
        run_dir.mkdir(parents=True, exist_ok=True)
        if job.kind == 'train':
            job.log_path = run_dir / 'job.log'
        else:
            (run_dir / 'jobs').mkdir(exist_ok=True)
            job.log_path = run_dir / 'jobs' / f'{int(job.created)}_{job.id}_{job.kind}.log'
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
               'HOME': os.environ.get('HOME', str(ROOT)),
               'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(ROOT)}
        try:
            log = open(job.log_path, 'ab')
            job.proc = subprocess.Popen(job.argv, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True, shell=False)
            log.close()
        except OSError as exc:
            job.state, job.error, job.ended = 'failed', str(exc), time.time()
            return
        job.pid = job.proc.pid
        job.state = 'running'
        job.started = time.time()

    # -- monitoring ------------------------------------------------------------

    def _loop(self):
        while not self._stop.wait(0.5):
            self.poll()

    def poll(self):
        changed, continuations = [], []
        with self.lock:
            for job in list(self.jobs.values()):
                if job.state != 'running':
                    continue
                if job.external:
                    if not pid_alive(job.pid):
                        st = read_json(self.runs_dir / job.run / 'status.json') or {}
                        job.state = st.get('state') if st.get('state') in (
                            'finished', 'failed', 'stopped') else 'stopped'
                        job.ended = time.time()
                        changed.append(job)
                    continue
                rc = job.proc.poll() if job.proc else None
                if rc is None:
                    continue
                job.returncode = rc
                job.ended = time.time()
                if job.kind == 'train':
                    st = read_json(self.runs_dir / job.run / 'status.json') or {}
                    state = st.get('state')
                    job.state = state if state in ('finished', 'failed', 'stopped') else (
                        'stopped' if job.term_sent else 'finished' if rc == 0 else 'failed')
                else:
                    job.state = 'stopped' if job.term_sent else (
                        'finished' if rc == 0 else 'failed')
                if job.state == 'failed' and job.log_path:
                    job.error = self.tail(job, 1).strip() or f'exit code {rc}'
                changed.append(job)
                if job.state == 'finished' and job.then:
                    continuations.append(job)
        for job in continuations:
            for c in job.then:
                try:
                    self.submit(c['kind'], c['args'])
                except job_schema.JobError as exc:
                    job.error = f'continuation rejected: {exc.detail}'
        for job in changed:
            self.on_change(job)
        if changed:
            self._schedule()

    # -- control -----------------------------------------------------------------

    def stop(self, jid: int, force: bool = False) -> Job:
        with self.lock:
            job = self.jobs.get(jid)
            if job is None:
                raise job_schema.JobError(f'job {jid} not found', 404, 'not_found')
            if job.state == 'queued':
                self.queue.remove(jid)
                job.state, job.ended = 'stopped', time.time()
            elif job.state == 'running':
                if force:
                    if job.term_sent is None or time.time() - job.term_sent < FORCE_AFTER_S:
                        wait = FORCE_AFTER_S - (time.time() - (job.term_sent or time.time()))
                        raise job_schema.JobError(
                            f'force kill is allowed {FORCE_AFTER_S:.0f} s after the first '
                            f'stop request ({wait:.0f} s left)', 409, 'too_early')
                    self._signal(job, signal.SIGKILL)
                else:
                    if job.term_sent is None:
                        job.term_sent = time.time()
                    self._signal(job, signal.SIGTERM)
        self.on_change(job)
        return job

    @staticmethod
    def _signal(job: Job, sig):
        try:
            os.kill(job.pid, sig)
        except (ProcessLookupError, PermissionError, TypeError):
            pass

    def tail(self, job: Job, lines: int = 200) -> str:
        if not job.log_path or not job.log_path.exists():
            return ''
        with open(job.log_path, 'rb') as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 256 * 1024))
            data = fh.read().decode('utf-8', 'replace')
        return '\n'.join(data.splitlines()[-lines:])

    def list(self) -> list[dict]:
        with self.lock:
            return [j.public() for j in sorted(self.jobs.values(), key=lambda j: -j.id)]

    def get(self, jid: int) -> Job | None:
        return self.jobs.get(jid)

    def snapshot(self) -> str:
        return json.dumps(self.list())
