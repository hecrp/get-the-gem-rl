"""Tolerant readers and atomic writers."""
import json
import threading

from game.replay_store import (append_jsonl, atomic_write_json, read_json, read_jsonl,
                               store_level, load_stored_level)


def test_jsonl_line_written_in_two_writes(tmp_path):
    path = tmp_path / 'a.jsonl'
    append_jsonl(path, [{'n': 1}])
    with open(path, 'a') as fh:          # a writer caught mid-line
        fh.write('{"n": 2, "x"')
        fh.flush()
        recs, off = read_jsonl(path)
        assert recs == [{'n': 1}]
        fh.write(': 3}\n')
    more, off2 = read_jsonl(path, off)
    assert more == [{'n': 2, 'x': 3}] and off2 > off
    assert read_jsonl(path, off2) == ([], off2)


def test_jsonl_broken_complete_line_is_flagged(tmp_path):
    path = tmp_path / 'b.jsonl'
    path.write_text('{"n": 1}\nnot json\n{"n": 3}\n')
    recs, _ = read_jsonl(path)
    assert recs[0] == {'n': 1} and '_broken' in recs[1] and recs[2] == {'n': 3}


def test_truncated_json_reads_as_default(tmp_path):
    path = tmp_path / 's.json'
    path.write_text('{"state": "runn')
    assert read_json(path, default='x') == 'x'
    assert read_json(tmp_path / 'missing.json') is None
    atomic_write_json(path, {'state': 'running'})
    assert read_json(path) == {'state': 'running'}
    assert not list(tmp_path.glob('.*.tmp'))


def test_atomic_writes_never_seen_partial(tmp_path):
    path = tmp_path / 'status.json'
    atomic_write_json(path, {'i': 0, 'pad': 'x' * 10000})
    stop = threading.Event()
    bad = []

    def reader():
        while not stop.is_set():
            data = read_json(path)
            if data is None or len(data['pad']) != 10000:
                bad.append(data)

    t = threading.Thread(target=reader)
    t.start()
    for i in range(300):
        atomic_write_json(path, {'i': i, 'pad': 'x' * 10000})
    stop.set()
    t.join()
    assert bad == []


def test_level_store_idempotent_and_validated(tmp_path):
    from game import Level
    level = Level.from_json('levels/level_008.json')
    h1 = store_level(tmp_path, level)
    h2 = store_level(tmp_path, level)
    assert h1 == h2 and len(list((tmp_path / '_levels').iterdir())) == 1
    assert load_stored_level(tmp_path, h1) == level
    assert load_stored_level(tmp_path, 'sha256:../../etc') is None
    assert load_stored_level(tmp_path, 'sha256:' + '0' * 64) is None
    text = (tmp_path / '_levels' / (h1.split(':')[1] + '.json')).read_text()
    assert ' ' not in text.split('"name"')[0] and json.loads(text)['name'] == level.name
