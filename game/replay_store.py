"""File formats shared by the recorders (``rl``) and readers (``dashboard``).

Standard library only, no wall clock (timestamps are supplied by callers):

* atomic JSON writes (write to a temporary file + ``os.replace``) so readers
  never see a truncated document (R-09);
* JSON Lines appends with one ``write`` per newline-terminated line, and a
  tolerant reader that ignores a last line without ``\\n`` (R-09);
* the content-addressed level store ``runs/_levels/<sha256>.json`` (R-03);
* per-run replay numbering under an exclusive file lock: numbers are
  monotonic, start at 1 and are never reused, even if replays are deleted
  (R-20).
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from .level import Level, canonical_json, level_hash

LEVELS_DIRNAME = '_levels'


# -- atomic / tolerant I/O ------------------------------------------------------

def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path, obj, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent))


def append_jsonl(path, records: Iterable[dict]) -> None:
    """Append records; each line is written with a single ``write`` call."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as fh:
        for rec in records:
            fh.write(json.dumps(rec, separators=(',', ':')) + '\n')
            fh.flush()


def read_json(path, default=None):
    """Parse a JSON file; ``default`` if it is missing or (mid-write) invalid."""
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def read_jsonl(path, offset: int = 0) -> tuple[list, int]:
    """Complete lines from byte ``offset`` on: ``(records, new_offset)``.

    A trailing line without ``\\n`` (a writer mid-``write``) is left for the
    next call; a complete line that is not valid JSON is returned as
    ``{"_broken": "<text>"}`` so callers can flag rather than hide it.
    """
    try:
        with open(path, 'rb') as fh:
            fh.seek(offset)
            data = fh.read()
    except OSError:
        return [], offset
    end = data.rfind(b'\n')
    if end < 0:
        return [], offset
    out = []
    for raw in data[:end].split(b'\n'):
        if not raw.strip():
            continue
        try:
            out.append(json.loads(raw))
        except ValueError:
            out.append({'_broken': raw.decode('utf-8', 'replace')[:200]})
    return out, offset + end + 1


# -- level store ------------------------------------------------------------------

def levels_dir(runs_dir) -> Path:
    return Path(runs_dir) / LEVELS_DIRNAME


def store_level(runs_dir, level: Level) -> str:
    """Write the level's canonical JSON once (idempotent); returns its hash."""
    digest = level_hash(level)
    path = levels_dir(runs_dir) / (digest.split(':', 1)[1] + '.json')
    if not path.exists():
        atomic_write_text(path, canonical_json(level.to_dict()))
    return digest


def load_stored_level(runs_dir, digest: str) -> Level | None:
    hexdigest = digest.split(':', 1)[-1]
    if not hexdigest or any(c not in '0123456789abcdef' for c in hexdigest):
        return None
    path = levels_dir(runs_dir) / f'{hexdigest}.json'
    try:
        return Level.from_dict(json.loads(path.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        return None


# -- replay numbering -----------------------------------------------------------

@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+') as fh:
        try:
            import fcntl
        except ImportError:                      # Windows (recommendation only)
            import msvcrt
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            return
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _max_number(replays_path: Path) -> int:
    records, _ = read_jsonl(replays_path)
    return max((r.get('n', 0) for r in records if isinstance(r.get('n'), int)), default=0)


def append_replays(run_dir, replays: list[dict]) -> list[dict]:
    """Number ``replays`` (``n``, ``id``) and append them to ``replays.jsonl``.

    Under ``replays.lock``: the next number comes from ``replays.seq`` (or, if
    missing, from the highest ``n`` in ``replays.jsonl``), so two concurrent
    writers never get the same number and a deleted replay's number is never
    handed out again. Returns the numbered records.
    """
    run_dir = Path(run_dir)
    run = run_dir.name
    seq_path = run_dir / 'replays.seq'
    with _locked(run_dir / 'replays.lock'):
        try:
            nxt = int(seq_path.read_text().strip())
        except (OSError, ValueError):
            nxt = _max_number(run_dir / 'replays.jsonl') + 1
        out = []
        for rep in replays:
            rep = dict(rep, n=nxt, id=f'{run}/{nxt:04d}', run=run)
            out.append(rep)
            nxt += 1
        atomic_write_text(seq_path, f'{nxt}\n')
        append_jsonl(run_dir / 'replays.jsonl', out)
    return out
