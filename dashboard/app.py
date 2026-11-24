"""FastAPI application: read API over ``runs/``, live events (SSE) and jobs.

Security (R-14): the server binds to loopback only; every request must carry a
loopback ``Host`` (defeats DNS rebinding); state-changing requests also need an
``Origin`` equal to the server itself and the ``X-Dashboard-Token`` printed at
start-up and embedded in the served page (defeats CSRF). Jobs are argument
lists validated against a whitelist (``job_schema``), never shell strings.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from game.level import canonical_json
from game.replay_store import read_json, read_jsonl
from rendering import palette

from . import __version__, job_schema
from .analytics import group_bands, heatmap, sweep_summary
from .jobs import JobManager, find_train_python
from .stats import lttb
from .store import REFERENCE_RUN, ROOT, Store

STATIC = Path(__file__).resolve().parent / 'static'
LOOPBACK = {'127.0.0.1', 'localhost', '::1', '[::1]'}
MAX_BODY = 64 * 1024
MAX_TRAJECTORY = 8 * 1024 * 1024
MAX_LIMIT = 500
DEFAULT_COLS = ('return', 'success', 'gem_rate', 'entropy', 'sps', 'eval_train', 'eval_test')


class ApiError(Exception):
    def __init__(self, status: int, code: str, detail: str = ''):
        self.status, self.code, self.detail = status, code, detail


def _host_ok(host: str | None) -> bool:
    if not host:
        return False
    name = host
    if host.startswith('['):
        name = host.split(']')[0] + ']'
    elif host.count(':') == 1:
        name = host.split(':')[0]
    return name in LOOPBACK


class Hub:
    """Fan-out of server-sent events to every connected client."""

    def __init__(self):
        self.clients: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.lock = threading.Lock()
        self.sent = 0

    def publish(self, event: str, data: dict):
        loop = self.loop
        if loop is None:
            return
        msg = f'event: {event}\ndata: {json.dumps(data, separators=(",", ":"))}\n\n'

        def push():
            for q in list(self.clients):
                if q.qsize() < 1000:
                    q.put_nowait(msg)
        try:
            loop.call_soon_threadsafe(push)
            self.sent += 1
        except RuntimeError:
            pass


def create_app(runs_dir, *, demo: bool = False, train_python: str | None = None,
               token: str | None = None, watch: bool = True, poll_s: float = 1.0,
               heartbeat_s: float = 15.0, find_python: bool = True) -> FastAPI:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    token = token or secrets.token_urlsafe(24)
    store = Store(runs_dir, demo=demo)
    hub = Hub()
    if find_python:
        python, python_note = find_train_python(train_python)
    else:
        python, python_note = train_python, 'given'
    jobs = JobManager(runs_dir, python,
                      on_change=lambda job: hub.publish('job.updated', job.public()))
    started = time.time()
    stop = threading.Event()
    caches: dict = {}

    def watcher():
        try:                                    # warm the generator's jump table
            from pcg import build_jump_table
            build_jump_table()
        except Exception:
            pass
        while not stop.wait(poll_s):
            try:
                changes = store.scan()
            except Exception as exc:                    # keep watching
                hub.publish('error', {'detail': f'scan: {exc}'})
                continue
            for name, what in changes['runs'].items():
                run = store.runs.get(name)
                hub.publish('run.updated', {'name': name, 'changed': what,
                                            'state': run.state() if run else None,
                                            'status': run.status if run else None})
            for e in changes['replays']:
                hub.publish('replay.added', {
                    'id': e['id'], 'run': e.get('run'), 'n': e.get('n'),
                    'policy': e.get('policy'), 'split': e.get('split'),
                    'source': e.get('source'), 'level': e.get('level'),
                    'outcome': e.get('outcome'), 'tags': e.get('tags')})
            for name in changes['removed']:
                hub.publish('run.updated', {'name': name, 'removed': True})

    @asynccontextmanager
    async def lifespan(app):
        hub.loop = asyncio.get_running_loop()
        threads = []
        if watch:
            jobs.start()
            t = threading.Thread(target=watcher, name='watcher', daemon=True)
            t.start()
            threads.append(t)
        yield
        stop.set()
        jobs.close()

    app = FastAPI(title='Get the Gem dashboard', version=__version__, lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store, app.state.jobs, app.state.hub, app.state.token = store, jobs, hub, token

    # -- errors -------------------------------------------------------------------

    def err(status, code, detail=''):
        return JSONResponse({'error': code, 'detail': detail}, status_code=status)

    @app.exception_handler(ApiError)
    async def _api_error(request, exc: ApiError):
        return err(exc.status, exc.code, exc.detail)

    @app.exception_handler(job_schema.JobError)
    async def _job_error(request, exc: job_schema.JobError):
        return err(exc.status, exc.code, exc.detail)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request, exc):
        code = {404: 'not_found', 405: 'method_not_allowed'}.get(exc.status_code, 'http_error')
        return err(exc.status_code, code, str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request, exc):
        return err(422, 'invalid_request', '; '.join(
            f"{'.'.join(str(x) for x in e.get('loc', ()))}: {e.get('msg')}" for e in exc.errors()))

    # -- security middleware ----------------------------------------------------------

    @app.middleware('http')
    async def security(request: Request, call_next):
        host = request.headers.get('host')
        if not _host_ok(host):
            return err(403, 'forbidden_host', 'only loopback Host headers are accepted')
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            origin = request.headers.get('origin')
            if origin != f'http://{host}':
                return err(403, 'forbidden_origin', 'Origin must be this server')
            if not secrets.compare_digest(request.headers.get('x-dashboard-token', ''), token):
                return err(403, 'forbidden_token', 'missing or wrong X-Dashboard-Token')
            length = request.headers.get('content-length')
            if length is not None and (not length.isdigit() or int(length) > MAX_BODY):
                return err(413, 'body_too_large', f'max {MAX_BODY} bytes')
        response = await call_next(request)
        response.headers.setdefault('X-Content-Type-Options', 'nosniff')
        response.headers.setdefault('Referrer-Policy', 'no-referrer')
        return response

    def etag_response(request: Request, payload, key: str, max_age='no-cache'):
        tag = '"' + hashlib.sha1(key.encode()).hexdigest()[:20] + '"'
        if request.headers.get('if-none-match') == tag:
            return Response(status_code=304, headers={'ETag': tag, 'Cache-Control': max_age})
        return JSONResponse(payload, headers={'ETag': tag, 'Cache-Control': max_age})

    def get_run(name: str):
        if name.startswith(('_', '.')) or '/' in name or '..' in name:
            raise ApiError(404, 'not_found', f'run {name!r} not found')
        run = store.get_run(name)
        if run is None:
            raise ApiError(404, 'not_found', f'run {name!r} not found')
        return run

    # -- page ------------------------------------------------------------------------

    @app.get('/', response_class=HTMLResponse)
    async def index():
        html = (STATIC / 'index.html').read_text(encoding='utf-8')
        html = html.replace('{{TOKEN}}', token).replace('{{VERSION}}', __version__)
        return HTMLResponse(html, headers={'Cache-Control': 'no-cache'})

    app.mount('/static', StaticFiles(directory=STATIC), name='static')

    # -- meta --------------------------------------------------------------------------

    @app.get('/api/health')
    async def health():
        return {'version': __version__, 'runs_dir': str(runs_dir), 'demo': demo,
                'torch': python is not None, 'torch_note': python_note,
                'train_python': python, 'started_at': started, 'now': time.time(),
                'runs': len(store.runs), 'sse_clients': len(hub.clients)}

    @app.get('/api/theme')
    async def theme():
        return palette.as_dict()

    # -- runs -------------------------------------------------------------------------

    @app.get('/api/runs')
    async def runs(request: Request):
        items = [r.summary() for r in sorted(store.runs.values(), key=lambda r: r.name)]
        return {'runs': items, 'demo': demo}

    @app.get('/api/runs/{name}')
    async def run_detail(name: str):
        run = get_run(name)
        return {**run.summary(), 'config_full': run.config, 'eval_raw': run.eval,
                'files': run.files(), 'sweeps_list': run.sweeps(),
                'observed_interval_s': run.observed_interval}

    @app.get('/api/runs/{name}/metrics')
    async def run_metrics(request: Request, name: str, cols: str = '', since: int = 0,
                          max_points: int = 1500, x: str = 'env_steps'):
        run = get_run(name)
        if not 3 <= max_points <= 20000:
            raise ApiError(422, 'invalid_request', 'max_points: 3..20000')
        if x not in ('env_steps', 'update', 't'):
            raise ApiError(422, 'invalid_request', 'x: env_steps, update or t')
        wanted = [c for c in (cols.split(',') if cols else DEFAULT_COLS) if c]
        key = f'{name}|{run.progress.stat}|{run.metrics.stat}|{wanted}|{since}|{max_points}|{x}'
        hit = caches.get(('metrics', name))
        if hit and hit[0] == key:
            return etag_response(request, hit[1], key)
        rows = run.rows()
        if since:
            rows = [r for r in rows if (r.get('update') or 0) > since]
        series = {}
        for col in wanted:
            pts = [(r.get(x), r.get(col)) for r in rows
                   if isinstance(r.get(col), (int, float)) and not isinstance(r.get(col), bool)
                   and isinstance(r.get(x), (int, float))]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            idx = lttb(xs, ys, max_points) if len(pts) > max_points else range(len(pts))
            series[col] = {'x': [xs[i] for i in idx], 'y': [ys[i] for i in idx], 'n': len(pts),
                           'decimated': len(pts) > max_points}
        last = rows[-1] if rows else (run.rows()[-1] if run.rows() else None)
        payload = {'run': name, 'x': x, 'series': series, 'rows': len(run.rows()),
                   'since': since, 'last_update': (last or {}).get('update'),
                   'source': 'metrics.jsonl' if run.metrics.rows else 'progress.csv',
                   'demo': run.demo, 'state': run.state()}
        caches[('metrics', name)] = (key, payload)
        return etag_response(request, payload, key)

    @app.get('/api/groups')
    async def groups(request: Request, cols: str = 'success,return'):
        wanted = [c for c in cols.split(',') if c][:6]
        by_group: dict[str, list] = {}
        for run in store.runs.values():
            by_group.setdefault(run.group, []).append(run)
        key = canonical_json([[g, sorted((r.name, str(r.progress.stat), str(r.metrics.stat))
                                         for r in rs)] for g, rs in sorted(by_group.items())]
                             + wanted)
        hit = caches.get('groups')
        if hit and hit[0] == key:
            return etag_response(request, hit[1], key)
        out = []
        for g, rs in sorted(by_group.items()):
            rs = sorted(rs, key=lambda r: r.name)
            item = {'group': g, 'runs': [r.name for r in rs],
                    'seeds': [r.config.get('seed') for r in rs],
                    'demo': any(r.demo for r in rs), 'cols': {}}
            for col in wanted:
                series = {}
                for r in rs:
                    pts = [(row['env_steps'], row[col]) for row in r.rows()
                           if isinstance(row.get(col), (int, float))
                           and isinstance(row.get('env_steps'), (int, float))]
                    series[r.name] = ([p[0] for p in pts], [p[1] for p in pts])
                item['cols'][col] = group_bands(series)
            out.append(item)
        payload = {'groups': out}
        caches['groups'] = (key, payload)
        return etag_response(request, payload, key)

    # -- replays -----------------------------------------------------------------------

    @app.get('/api/replays')
    async def replays(run: str = '', split: str = '', policy: str = '', outcome: str = '',
                      level: str = '', source: str = '', tag: str = '', after: str = '',
                      limit: int = 50, order: str = 'number', seed: int = 0):
        if not 1 <= limit <= MAX_LIMIT:
            raise ApiError(422, 'invalid_request', f'limit: 1..{MAX_LIMIT}')
        if order not in ('number', 'latest', 'random'):
            raise ApiError(422, 'invalid_request', 'order: number, latest or random')
        return store.index.query(run=run or None, split=split or None, policy=policy or None,
                                 outcome=outcome or None, level=level or None,
                                 source=source or None, tag=tag or None, after=after or None,
                                 limit=limit, order=order, seed=seed)

    def find_replay(run: str, n: str) -> dict:
        rec = store.replay(run, n)
        if rec is None:
            raise ApiError(404, 'not_found', f'replay {run}/{n} not found')
        return rec

    @app.get('/api/replays/{run}/{n}')
    async def replay(run: str, n: str):
        rec = find_replay(run, n)
        entry = store.entry(rec['id']) or {}
        return {**rec, 'broken': entry.get('broken'), 'demo': entry.get('demo', demo)}

    @app.get('/api/replays/{run}/{n}/trajectory')
    async def trajectory(request: Request, run: str, n: str, cam: str = 'none'):
        if cam not in ('none', 'follow'):
            raise ApiError(422, 'invalid_request', 'cam: none or follow')
        rec = find_replay(run, n)
        try:
            body, meta = await asyncio.to_thread(store.trajectory, rec, cam)
        except LookupError as exc:
            store.mark_broken(rec['id'], str(exc))
            raise ApiError(409, 'level_unresolvable', str(exc))
        except (ValueError, TypeError) as exc:
            store.mark_broken(rec['id'], str(exc))
            raise ApiError(422, 'replay_broken', str(exc))
        if meta['raw_bytes'] > MAX_TRAJECTORY:
            raise ApiError(413, 'trajectory_too_large',
                           f"{meta['raw_bytes']} bytes > {MAX_TRAJECTORY}")
        tag = '"' + hashlib.sha1(body).hexdigest()[:20] + '"'
        headers = {'ETag': tag, 'Cache-Control': 'no-cache', 'Vary': 'Accept-Encoding'}
        if request.headers.get('if-none-match') == tag:
            return Response(status_code=304, headers=headers)
        if 'gzip' in request.headers.get('accept-encoding', ''):
            return Response(body, media_type='application/json',
                            headers={**headers, 'Content-Encoding': 'gzip'})
        return Response(gzip.decompress(body), media_type='application/json', headers=headers)

    # -- levels ------------------------------------------------------------------------

    @app.get('/api/levels/builtin')
    async def levels_builtin():
        out = []
        for rec in store.reference:
            level, _, _ = store.resolve(rec['level']['ref'], rec['level']['hash'])
            out.append({'n': rec['n'], 'name': level.name, 'width': level.width,
                        'height': level.height, 'win_condition': level.win_condition,
                        'max_steps': level.max_steps, 'hash': rec['level']['hash'],
                        'platforms': len(level.platforms), 'hazards': len(level.hazards),
                        'has_exit': level.exit is not None,
                        'reference_steps': rec['outcome']['engine_steps'],
                        'reference_id': rec['id'], 'ref': rec['level']['ref']})
        return {'levels': out}

    def check_ref(ref):
        if not isinstance(ref, dict) or ref.get('kind') not in ('builtin', 'pcg', 'custom'):
            raise ApiError(422, 'invalid_request', 'ref: expected a LevelRef object')
        if ref['kind'] == 'pcg':
            from pcg import PRESETS
            seed = ref.get('seed')
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
                raise ApiError(422, 'invalid_request', 'ref.seed: integer in [0, 2^63)')
            if ref.get('config') is None and ref.get('preset') not in PRESETS:
                raise ApiError(422, 'invalid_request', 'ref: preset or config required')
        return ref

    @app.get('/api/levels/by-ref')
    async def level_by_ref(ref: str, hash: str = ''):
        try:
            data = json.loads(ref)
        except ValueError:
            raise ApiError(422, 'invalid_request', 'ref: invalid JSON')
        check_ref(data)
        try:
            level, drift, _ = await asyncio.to_thread(store.resolve, data, hash or None)
        except Exception as exc:                       # generator errors
            raise ApiError(422, 'invalid_request', f'cannot build level: {exc}')
        if level is None:
            raise ApiError(404, 'not_found', 'level not resolvable')
        return store.level_payload(level, data, drift)

    @app.get('/api/levels/by-hash/{digest}')
    async def level_by_hash(digest: str):
        if not digest.startswith('sha256:'):
            digest = 'sha256:' + digest
        level = await asyncio.to_thread(store.level_by_hash, digest)
        if level is None:
            raise ApiError(404, 'not_found', f'level {digest} not in the store')
        return store.level_payload(level, None, False)

    @app.get('/api/levels/expressive')
    async def expressive(preset: str = 'tiny', start: int = 0, n: int = 100):
        from pcg import PRESETS
        if preset not in PRESETS:
            raise ApiError(422, 'invalid_request', 'preset: unknown')
        cap = {'tiny': 500, 'small': 200, 'wide': 200, 'tall': 200}.get(preset, 40)
        if not 1 <= n <= cap or start < 0:
            raise ApiError(422, 'invalid_request', f'n: 1..{cap} for {preset}')
        key = ('expressive', preset, start, n)
        if key in caches:
            return caches[key]

        def compute():
            from pcg import generate_level, metrics
            pts = []
            for seed in range(start, start + n):
                m = metrics(generate_level(seed, PRESETS[preset]))
                pts.append({'seed': seed, **{k: getattr(m, k) for k in m.__dataclass_fields__}})
            return pts
        payload = {'preset': preset, 'start': start, 'n': n,
                   'points': await asyncio.to_thread(compute)}
        caches[key] = payload
        return payload

    @app.get('/api/levels/{n}/reference')
    async def level_reference(n: int):
        rec = store.replay(REFERENCE_RUN, str(n))
        if rec is None:
            raise ApiError(404, 'not_found', f'no reference solution for level {n}')
        return rec

    # -- sweeps -------------------------------------------------------------------------

    @app.get('/api/sweeps/{run}')
    async def sweeps(run: str):
        return {'run': run, 'sweeps': get_run(run).sweeps()}

    def sweep_lines(run, tag):
        r = get_run(run)
        if not job_schema.NAME_RE.match(tag):
            raise ApiError(404, 'not_found', 'sweep not found')
        path = r.dir / 'sweeps' / f'{tag}.jsonl'
        if not path.exists():
            raise ApiError(404, 'not_found', f'sweep {run}/{tag} not found')
        st = path.stat()
        key = ('sweep', run, tag, st.st_mtime_ns, st.st_size)
        hit = caches.get(('sweep_lines', run, tag))
        if hit and hit[0] == key:
            return hit[1], key
        lines, _ = read_jsonl(path)
        caches[('sweep_lines', run, tag)] = (key, lines)
        return lines, key

    @app.get('/api/sweeps/{run}/{tag}/summary')
    async def sweep_summary_route(request: Request, run: str, tag: str):
        lines, key = sweep_lines(run, tag)
        hit = caches.get(('summary', run, tag))
        if not hit or hit[0] != key:
            meta = read_json(get_run(run).dir / 'sweeps' / f'{tag}.meta.json', {}) or {}
            payload = {'run': run, 'tag': tag, 'meta': meta, 'demo': get_run(run).demo,
                       **await asyncio.to_thread(sweep_summary, lines)}
            hit = caches[('summary', run, tag)] = (key, payload)
        return etag_response(request, hit[1], str(key))

    @app.get('/api/sweeps/{run}/{tag}/heatmap')
    async def sweep_heatmap(run: str, tag: str, level: str = '', layer: str = 'visits'):
        if layer not in ('visits', 'deaths'):
            raise ApiError(422, 'invalid_request', 'layer: visits or deaths')
        lines, key = sweep_lines(run, tag)
        if not level:
            # Default: the most played level (visits) or the deadliest (deaths).
            counts = {}
            for x in lines:
                if layer == 'deaths' and not x.get('death_xy'):
                    continue
                counts[x.get('level_hash')] = counts.get(x.get('level_hash'), 0) + 1
            if not counts:
                counts = {x.get('level_hash'): 1 for x in lines}
            level = max(counts, key=counts.get) if counts else ''
        ck = ('heat', run, tag, level, layer)
        hit = caches.get(ck)
        if hit and hit[0] == key:
            return hit[1]
        sample = next((x for x in lines if x.get('level_hash') == level), None)
        if sample is None:
            raise ApiError(404, 'not_found', 'no episode on that level in this sweep')
        lv, drift, _ = store.resolve(sample.get('level_ref'), level)
        if lv is None:
            raise ApiError(409, 'level_unresolvable', 'level not resolvable')
        payload = await asyncio.to_thread(heatmap, lines, lv, level, layer)
        payload['level'] = store.level_payload(lv, sample.get('level_ref'), drift)
        payload['demo'] = get_run(run).demo
        caches[ck] = (key, payload)
        return payload

    # -- jobs ----------------------------------------------------------------------------

    @app.get('/api/jobs')
    async def jobs_list():
        return {'jobs': jobs.list(), 'enabled': python is not None, 'note': python_note,
                'limits': jobs.limits, 'queue': list(jobs.queue)}

    @app.get('/api/jobs/schema')
    async def jobs_schema():
        return job_schema.describe()

    def get_job(jid: int):
        job = jobs.get(jid)
        if job is None:
            raise ApiError(404, 'not_found', f'job {jid} not found')
        return job

    @app.get('/api/jobs/{jid}')
    async def job_detail(jid: int):
        return get_job(jid).public()

    @app.get('/api/jobs/{jid}/log')
    async def job_log(jid: int, tail: int = 200):
        if not 1 <= tail <= 5000:
            raise ApiError(422, 'invalid_request', 'tail: 1..5000')
        return PlainTextResponse(jobs.tail(get_job(jid), tail),
                                 headers={'Cache-Control': 'no-cache'})

    @app.post('/api/jobs', status_code=202)
    async def jobs_submit(request: Request):
        raw = await request.body()
        if len(raw) > MAX_BODY:
            raise ApiError(413, 'body_too_large', f'max {MAX_BODY} bytes')
        try:
            body = json.loads(raw or b'{}')
        except ValueError:
            raise ApiError(422, 'invalid_request', 'body: invalid JSON')
        if not isinstance(body, dict) or set(body) - {'kind', 'args', 'then'}:
            raise ApiError(422, 'invalid_request', 'body: {kind, args, then?}')
        job = await asyncio.to_thread(jobs.submit, body.get('kind'), body.get('args', {}),
                                      body.get('then'))
        return JSONResponse(job.public(), status_code=202)

    @app.delete('/api/jobs/{jid}')
    async def jobs_stop(jid: int, force: int = 0):
        return jobs.stop(jid, force=bool(force)).public()

    # -- events --------------------------------------------------------------------------

    @app.get('/api/events')
    async def events(request: Request):
        q: asyncio.Queue = asyncio.Queue()
        hub.clients.add(q)

        async def stream():
            try:
                yield f'event: hello\ndata: {json.dumps({"version": __version__, "demo": demo})}\n\n'
                last = time.monotonic()
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.wait_for(q.get(), timeout=1.0)
                        yield msg
                    except asyncio.TimeoutError:
                        pass
                    if time.monotonic() - last >= heartbeat_s:
                        last = time.monotonic()
                        yield f'event: heartbeat\ndata: {json.dumps({"t": time.time()})}\n\n'
            finally:
                hub.clients.discard(q)

        return StreamingResponse(stream(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    return app
