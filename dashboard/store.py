"""The file system under ``runs/`` as the dashboard's only source of truth.

Everything here is a rebuildable cache: run metadata re-read when a file's
``(mtime, size)`` changes, ``progress.csv``/``metrics.jsonl``/``replays.jsonl``
read incrementally from the last byte offset, levels resolved (and their
drift detected) through ``game``/``pcg`` and the content-addressed store, and
trajectories rebuilt with ``game.replay.simulate``. Never imports ``rl``,
``torch``, ``numpy`` or ``pygame``.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import random
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from game import Level
from game.camera import compute_viewport
from game.level import canonical_json, level_hash
from game.replay import reference_replays, rle_decode, simulate_replay
from game.replay_store import load_stored_level, read_json, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
BUILTIN_DIR = ROOT / 'levels'
SOLUTIONS_DIR = BUILTIN_DIR / 'solutions'
REFERENCE_RUN = '_reference'

# Config keys that do not define an experiment (R-16 default grouping).
NON_EXPERIMENT_KEYS = {
    'seed', 'name', 'threads', 'device', 'log_every', 'runs_dir', 'resume',
    'ckpt_every', 'ckpt_keep', 'ckpt_thin', 'replay_every', 'replay_n', 'group',
    'eval_envs', 'demo', '_explicit',
}
PROGRESS_FLOATS = ('return', 'success', 'gem_rate', 'pg_loss', 'v_loss', 'entropy',
                   'approx_kl', 'eval_train', 'eval_test')


def _stat(path: Path):
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def parse_iso(text):
    try:
        return datetime.strptime(text, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def group_of(config: dict, status: dict | None = None) -> str:
    """``--group`` if set, else a hash of the configuration without seed/name."""
    group = config.get('group') or (status or {}).get('group')
    if group:
        return str(group)
    exp = {k: v for k, v in config.items() if k not in NON_EXPERIMENT_KEYS}
    return 'cfg-' + hashlib.sha1(canonical_json(exp).encode()).hexdigest()[:10]


def _num(text):
    if text is None or text == '':
        return None
    try:
        v = float(text)
    except ValueError:
        return None
    return int(v) if v.is_integer() and 'e' not in text.lower() and '.' not in text else v


class IncrementalRows:
    """Rows of ``progress.csv`` or ``metrics.jsonl`` read by byte offset; the
    cache resets if the file shrinks (rewritten) and ignores a partial last line."""

    def __init__(self, path: Path, kind: str):
        self.path, self.kind = path, kind
        self.rows: list[dict] = []
        self.offset = 0
        self.header = None
        self.stat = None

    def refresh(self) -> bool:
        st = _stat(self.path)
        if st == self.stat:
            return False
        if st is None or (self.stat is not None and st[1] < self.offset):
            self.rows, self.offset, self.header = [], 0, None
        self.stat = st
        if st is None:
            return True
        if self.kind == 'jsonl':
            recs, self.offset = read_jsonl(self.path, self.offset)
            self.rows += [r for r in recs if '_broken' not in r]
            return True
        try:
            with open(self.path, 'rb') as fh:
                fh.seek(self.offset)
                data = fh.read()
        except OSError:
            return True
        end = data.rfind(b'\n')
        if end < 0:
            return True
        text = data[:end].decode('utf-8', 'replace')
        self.offset += end + 1
        for row in csv.reader(io.StringIO(text)):
            if not row:
                continue
            if self.header is None:
                self.header = row
                continue
            rec = {}
            for k, v in zip(self.header, row):
                rec[k] = _num(v)
            self.rows.append(rec)
        return True


class Run:
    """Cached view of one run folder."""

    def __init__(self, store: 'Store', name: str):
        self.store, self.name = store, name
        self.dir = store.runs_dir / name
        self._files: dict[str, tuple] = {}
        self.config: dict = {}
        self.status: dict | None = None
        self.eval: dict | None = None
        self.progress = IncrementalRows(self.dir / 'progress.csv', 'csv')
        self.metrics = IncrementalRows(self.dir / 'metrics.jsonl', 'jsonl')
        self.replay_offset = 0
        self.replay_stat = None
        self._prev_updated = None
        self.observed_interval = None

    def _changed(self, fname) -> bool:
        st = _stat(self.dir / fname)
        if self._files.get(fname, ()) != st:
            self._files[fname] = st
            return True
        return False

    def refresh(self) -> set:
        changed = set()
        if self._changed('config.json'):
            self.config = read_json(self.dir / 'config.json', {}) or {}
            changed.add('config')
        if self._changed('status.json'):
            st = read_json(self.dir / 'status.json')
            if st is not None:
                t = parse_iso(st.get('updated_at'))
                if self._prev_updated is not None and t is not None and t > self._prev_updated:
                    self.observed_interval = t - self._prev_updated
                self._prev_updated = t
                self.status = st
            changed.add('status')
        if self._changed('eval.json'):
            self.eval = read_json(self.dir / 'eval.json')
            changed.add('eval')
        if self.progress.refresh():
            changed.add('progress')
        if self.metrics.refresh():
            changed.add('metrics')
        return changed

    # -- derived ------------------------------------------------------------

    @property
    def demo(self) -> bool:
        return bool(self.config.get('demo')) or self.store.demo

    @property
    def group(self) -> str:
        return group_of(self.config, self.status)

    def state(self) -> dict:
        """State with dead detection (R-10): a ``running`` run whose pid is gone
        or whose heartbeat is older than 3x the observed interval (>= 120 s)."""
        st = self.status
        if st is None:
            has_model = (self.dir / 'model.pt').exists()
            return {'state': 'finished' if has_model or self.eval else 'unknown',
                    'legacy': True}
        state = st.get('state', 'unknown')
        out = {'state': state, 'legacy': False}
        if state == 'running':
            interval = st.get('log_interval_s') or self.observed_interval or 0
            limit = max(120.0, 3 * float(interval))
            updated = parse_iso(st.get('updated_at'))
            age = None if updated is None else time.time() - updated
            out['heartbeat_age_s'] = age
            if not pid_alive(st.get('pid')):
                out.update(state='dead', reason='pid not running')
            elif age is not None and age > limit:
                out.update(state='dead', reason=f'no heartbeat for {age:.0f} s')
        return out

    def rows(self) -> list[dict]:
        return self.metrics.rows if self.metrics.rows else self.progress.rows

    def files(self) -> dict:
        def exists(p):
            return (self.dir / p).exists()
        ckpts = sorted(x.name for x in (self.dir / 'ckpt').glob('update_*.pt')) \
            if exists('ckpt') else []
        return {'config': exists('config.json'), 'status': exists('status.json'),
                'progress': exists('progress.csv'), 'metrics': exists('metrics.jsonl'),
                'eval': exists('eval.json'), 'model': exists('model.pt'),
                'replays': exists('replays.jsonl'), 'job_log': exists('job.log'),
                'ckpts': ckpts}

    def sweeps(self) -> list[dict]:
        out = []
        for meta in sorted((self.dir / 'sweeps').glob('*.meta.json')):
            m = read_json(meta, {}) or {}
            tag = meta.name[:-len('.meta.json')]
            out.append({'tag': tag, 'n': m.get('n'), 'policy': m.get('policy'),
                        'ckpt': m.get('ckpt'), 'created': m.get('created'),
                        'seed_range': m.get('seed_range'), 'levels': m.get('levels')})
        return out

    def summary(self) -> dict:
        cfg = self.config
        rows = self.rows()
        last = rows[-1] if rows else None
        ev = self.eval or {}
        return {
            'name': self.name, 'group': self.group, **self.state(),
            'status': self.status, 'demo': self.demo,
            'config': {k: cfg.get(k) for k in (
                'preset', 'levels', 'num_levels', 'start_seed', 'total_steps',
                'distance_shaping', 'gem_reward', 'frame_skip', 'seed', 'no_goal_vector',
                'ckpt_every', 'replay_every')},
            'last': last, 'points': len(rows),
            'eval': summarize_eval(ev, cfg),
            'replays': self.store.index.count(self.name),
            'sweeps': len(list((self.dir / 'sweeps').glob('*.meta.json'))),
            'has_ckpts': bool(list((self.dir / 'ckpt').glob('update_*.pt'))),
            'only_final_policy': not (self.dir / 'ckpt').is_dir(),
        }


def summarize_eval(ev: dict, cfg: dict) -> dict | None:
    """``eval.json`` as ``{policy: {set: {k, n, p, lo, hi}}}`` with Wilson CIs."""
    from .stats import wilson
    if not ev or ev.get('partial'):
        return {'partial': True, **ev} if ev else None
    out = {}
    for policy in ('sampled', 'greedy'):
        block = ev.get(policy)
        if not isinstance(block, dict):
            continue
        out[policy] = {}
        for key, res in block.items():
            if not isinstance(res, dict) or 'success_rate' not in res:
                continue
            n = int(res.get('episodes', 0))
            k = round(res['success_rate'] * n)
            w = wilson(k, n)
            w['gem'] = wilson(round(res.get('gem_rate', 0) * n), n)['p']
            w['mean_return'] = res.get('mean_return')
            out[policy][key] = w
    out['train_seconds'] = ev.get('train_seconds')
    return out


# -- replay index ------------------------------------------------------------------

class ReplayIndex:
    """In-memory, rebuildable index of every replay (without actions)."""

    def __init__(self, store: 'Store'):
        self.store = store
        self.entries: dict[str, dict] = OrderedDict()
        self.by_run: dict[str, list[str]] = {}
        self.seq = 0
        self.lock = threading.Lock()

    def count(self, run: str) -> int:
        return len(self.by_run.get(run, ()))

    def _entry(self, rec: dict, run: str) -> dict:
        broken = None
        if '_broken' in rec:
            broken = 'invalid JSON line'
        else:
            try:
                for _ in rle_decode(rec.get('actions', [])):
                    pass
            except (ValueError, TypeError) as exc:
                broken = f'actions: {exc}'
            lvl = rec.get('level') or {}
            if not broken and not lvl.get('hash'):
                broken = 'level: no hash'
            if not broken and (lvl.get('ref') or {}).get('kind') not in ('builtin', 'pcg'):
                if not (self.store.runs_dir / '_levels' /
                        (str(lvl.get('hash', '')).split(':')[-1] + '.json')).exists():
                    broken = 'level: hash not in the level store'
        self.seq += 1
        entry = {k: v for k, v in rec.items() if k != 'actions'}
        entry.setdefault('run', run)
        entry['_seq'] = self.seq
        entry['broken'] = broken
        entry['demo'] = self.store.demo or 'demo' in (rec.get('tags') or [])
        entry['action_runs'] = len(rec.get('actions') or [])
        if 'id' not in entry:
            entry['id'] = f"{run}/{entry.get('n', self.seq):04d}" if isinstance(
                entry.get('n'), int) else f'{run}/x{self.seq}'
        return entry

    def add_reference(self):
        for rec in self.store.reference:
            e = self._entry(rec, REFERENCE_RUN)
            self.entries[e['id']] = e
            self.by_run.setdefault(REFERENCE_RUN, []).append(e['id'])

    def refresh_run(self, run: Run) -> list[dict]:
        """Index new lines of ``replays.jsonl``; returns the new entries."""
        path = run.dir / 'replays.jsonl'
        st = _stat(path)
        if st == run.replay_stat:
            return []
        if st is None or (run.replay_stat is not None and st[1] < run.replay_offset):
            with self.lock:
                for rid in self.by_run.pop(run.name, []):
                    self.entries.pop(rid, None)
            run.replay_offset = 0
        run.replay_stat = st
        if st is None:
            return []
        recs, run.replay_offset = read_jsonl(path, run.replay_offset)
        new = []
        with self.lock:
            for rec in recs:
                e = self._entry(rec, run.name)
                self.entries[e['id']] = e
                self.by_run.setdefault(run.name, []).append(e['id'])
                new.append(e)
        return new

    def query(self, *, run=None, split=None, policy=None, outcome=None, level=None,
              source=None, tag=None, after=None, limit=50, order='number', seed=0,
              include_broken=True):
        items = list(self.entries.values())
        runs = set(run.split(',')) if run else None

        def ok(e):
            if runs and e.get('run') not in runs:
                return False
            if split and e.get('split') != split:
                return False
            if policy and not str(e.get('policy', '')).startswith(policy):
                return False
            if source and (e.get('source') or {}).get('kind') != source:
                return False
            if tag and tag not in (e.get('tags') or []):
                return False
            if outcome:
                o = e.get('outcome') or {}
                if outcome not in (o.get('result'), o.get('end_reason')):
                    return False
            if level:
                lv = e.get('level') or {}
                ref = lv.get('ref') or {}
                keys = {lv.get('hash'), lv.get('name')}
                if ref.get('kind') == 'builtin':
                    keys.add(f"builtin:{ref.get('n')}")
                if ref.get('kind') == 'pcg':
                    keys.add(f"pcg:{ref.get('preset')}:{ref.get('seed')}")
                if level not in keys:
                    return False
            return include_broken or not e.get('broken')

        items = [e for e in items if ok(e)]
        if order == 'latest':
            items.sort(key=lambda e: -e['_seq'])
        elif order == 'random':
            items.sort(key=lambda e: e['_seq'])
            random.Random(seed).shuffle(items)
        else:
            items.sort(key=lambda e: (e.get('run', ''), e.get('n') if isinstance(
                e.get('n'), int) else 0))
        total = len(items)
        if after:
            ids = [e['id'] for e in items]
            items = items[ids.index(after) + 1:] if after in ids else []
        page = items[:limit]
        nxt = page[-1]['id'] if len(items) > limit else None
        return {'total': total, 'items': [_public(e) for e in page], 'next': nxt}


def _public(e: dict) -> dict:
    return {k: v for k, v in e.items() if not k.startswith('_')}


# -- the store ----------------------------------------------------------------------

class LRU:
    """Byte-bounded LRU for gzip trajectories (R-21)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.data: OrderedDict = OrderedDict()
        self.size = 0
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            v = self.data.get(key)
            if v is not None:
                self.data.move_to_end(key)
            return v

    @staticmethod
    def _size(value) -> int:
        return len(value[0]) if isinstance(value, tuple) else len(value)

    def put(self, key, value):
        with self.lock:
            if key in self.data:
                self.size -= self._size(self.data.pop(key))
            self.data[key] = value
            self.size += self._size(value)
            while self.size > self.max_bytes and self.data:
                _, old = self.data.popitem(last=False)
                self.size -= self._size(old)


class Store:
    def __init__(self, runs_dir, demo: bool = False, cache_bytes: int = 256 << 20):
        self.runs_dir = Path(runs_dir)
        self.demo = demo
        self.runs: dict[str, Run] = {}
        self.index = ReplayIndex(self)
        self.reference = reference_replays(SOLUTIONS_DIR, BUILTIN_DIR)
        self.index.add_reference()
        self.trajectories = LRU(cache_bytes)
        self._levels: OrderedDict = OrderedDict()
        self._level_lock = threading.Lock()
        self.lock = threading.RLock()
        self.scan()

    # -- runs -----------------------------------------------------------------

    def run_names(self) -> list[str]:
        try:
            names = [p.name for p in self.runs_dir.iterdir()
                     if p.is_dir() and not p.name.startswith(('_', '.'))]
        except OSError:
            names = []
        return sorted(names)

    def scan(self) -> dict:
        """Refresh every run; returns ``{"runs": {name: changed}, "replays": [...]}``."""
        out = {'runs': {}, 'replays': [], 'removed': []}
        with self.lock:
            names = self.run_names()
            for gone in set(self.runs) - set(names):
                del self.runs[gone]
                for rid in self.index.by_run.pop(gone, []):
                    self.index.entries.pop(rid, None)
                out['removed'].append(gone)
            for name in names:
                run = self.runs.get(name)
                if run is None:
                    run = self.runs[name] = Run(self, name)
                changed = run.refresh()
                new = self.index.refresh_run(run)
                if new:
                    changed.add('replays')
                    out['replays'] += new
                sweeps = run.dir / 'sweeps'
                st = _stat(sweeps)
                if st is not None:
                    key = tuple(sorted((p.name, _stat(p)) for p in sweeps.iterdir()))
                    if getattr(run, '_sweeps_key', None) != key:
                        run._sweeps_key = key
                        changed.add('sweeps')
                if changed:
                    out['runs'][name] = sorted(changed)
        return out

    def get_run(self, name: str) -> Run | None:
        with self.lock:
            run = self.runs.get(name)
            if run is None and name in self.run_names():
                run = self.runs[name] = Run(self, name)
                run.refresh()
                self.index.refresh_run(run)
            return run

    # -- levels -------------------------------------------------------------------

    def _cache_level(self, key, value):
        with self._level_lock:
            self._levels[key] = value
            if len(self._levels) > 512:
                self._levels.popitem(last=False)

    def resolve(self, ref: dict | None, digest: str | None = None):
        """``(level, drift, source)`` for a ``LevelRef`` and expected hash (R-03).

        The ref is regenerated (built-in file or ``pcg.generate_level``); if its
        hash differs from ``digest`` (or it cannot be regenerated) the snapshot
        in ``runs/_levels/`` is used and ``drift`` is True. ``level`` is None if
        nothing matches.
        """
        key = (canonical_json(ref) if ref else None, digest)
        with self._level_lock:
            hit = self._levels.get(key)
        if hit is not None:
            return hit
        level, source = None, None
        try:
            level = self._regenerate(ref) if ref else None
            source = 'regenerated'
        except (ValueError, TypeError, KeyError, IndexError, OSError):
            level = None
        drift = False
        if level is not None and digest and level_hash(level) != digest:
            level, drift = None, True
        if level is None and digest:
            snap = load_stored_level(self.runs_dir, digest)
            if snap is not None:
                level, source = snap, 'snapshot'
                drift = drift or bool(ref and ref.get('kind') in ('builtin', 'pcg'))
        result = (level, drift, source)
        self._cache_level(key, result)
        return result

    def _regenerate(self, ref: dict) -> Level | None:
        kind = ref.get('kind')
        if kind == 'builtin':
            n = int(ref['n'])
            if not 1 <= n <= 9:
                raise ValueError('builtin: 1..9')
            return Level.from_json(BUILTIN_DIR / f'level_{n:03d}.json')
        if kind == 'pcg':
            from pcg import GENERATOR_VERSION, GenConfig, generate_level
            if ref.get('generator_version', GENERATOR_VERSION) != GENERATOR_VERSION:
                return None
            config = ref.get('config')
            if config is None:
                from pcg import PRESETS
                cfg = PRESETS[ref['preset']]
            else:
                cfg = GenConfig(**config)
            seed = int(ref['seed'])
            if seed < 0:
                raise ValueError('seed')
            return generate_level(seed, cfg)
        if kind == 'custom':
            return None
        raise ValueError(f'unknown level kind {kind!r}')

    def level_by_hash(self, digest: str) -> Level | None:
        for rec in self.reference:
            if rec['level']['hash'] == digest:
                return self.resolve(rec['level']['ref'], digest)[0]
        return load_stored_level(self.runs_dir, digest)

    # -- replays ------------------------------------------------------------------

    def replay(self, run: str, n: str) -> dict | None:
        """Full replay record (with actions)."""
        if run == REFERENCE_RUN:
            for rec in self.reference:
                if n in (f"level_{rec['n']:03d}", str(rec['n']), f"{rec['n']:04d}"):
                    return rec
            return None
        r = self.get_run(run)
        if r is None:
            return None
        try:
            num = int(n)
        except ValueError:
            return None
        recs, _ = read_jsonl(r.dir / 'replays.jsonl')
        for rec in recs:
            if rec.get('n') == num:
                return rec
        return None

    def entry(self, rid: str) -> dict | None:
        return self.index.entries.get(rid)

    def mark_broken(self, rid: str, why: str):
        e = self.index.entries.get(rid)
        if e is not None:
            e['broken'] = why

    def trajectory(self, rec: dict, cam: str = 'none') -> tuple[bytes, dict]:
        """Gzip JSON of the trajectory (spec 13 §91.5) and a small header dict."""
        rid = rec['id']
        lvl = rec.get('level') or {}
        key = (rid, lvl.get('hash'), cam, canonical_json(rec.get('actions', [])))
        hit = self.trajectories.get(key)
        if hit is not None:
            return hit
        level, drift, source = self.resolve(lvl.get('ref'), lvl.get('hash'))
        if level is None:
            raise LookupError('level not resolvable (no matching snapshot)')
        traj = simulate_replay(rec, level, record=True)
        body = {'id': rid, 'engine_steps': traj.engine_steps,
                'env_steps': traj.env_steps,
                'frame_skip': (rec.get('env') or {}).get('frame_skip', 1),
                'world': [level.width, level.height], 'level_hash': level_hash(level),
                'drift': drift, 'level_source': source,
                'result': traj.result, 'end_reason': traj.end_reason,
                'xy': list(traj.xy), 'flags': list(traj.flags), 'events': list(traj.events)}
        if cam == 'follow':
            xy = traj.xy
            camv = []
            for i in range(0, len(xy), 2):
                vp = compute_viewport(level.width, level.height, xy[i], xy[i + 1])
                camv += (int(vp.x), int(vp.y))
            body['cam'] = camv
            body['view'] = [240, 160]
        raw = json.dumps(body, separators=(',', ':')).encode()
        packed = (gzip.compress(raw, 6), {'raw_bytes': len(raw), 'drift': drift})
        self.trajectories.put(key, packed)
        return packed

    def level_payload(self, level: Level, ref=None, drift=False) -> dict:
        """Geometry for the client: ``to_dict`` plus resolved rectangles."""
        data = {
            'hash': level_hash(level), 'name': level.name, 'width': level.width,
            'height': level.height, 'win_condition': level.win_condition,
            'max_steps': level.max_steps, 'player_spawn': list(level.player_spawn),
            'gem': [level.gem.x, level.gem.y, level.gem.w, level.gem.h],
            'exit': None if level.exit is None else [level.exit.x, level.exit.y,
                                                     level.exit.w, level.exit.h],
            'platforms': [[p.x, p.y, p.w, p.h] for p in level.platforms],
            'hazards': [[h.x, h.y, h.w, h.h] for h in level.hazards],
            'ref': ref, 'drift': drift, 'source': level.to_dict(),
        }
        if level._metadata.get('generator') and level.tiles is not None:
            from pcg import metrics
            m = metrics(level)
            data['metrics'] = {k: getattr(m, k) for k in m.__dataclass_fields__}
        data['source'].get('metadata', {}).pop('waypoints', None)
        return data
