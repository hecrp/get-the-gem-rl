"""Dashboard read API."""
import gzip
import json

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')

from game.replay import simulate_replay  # noqa: E402
from game.replay_store import append_replays, read_jsonl  # noqa: E402

from dashboard_helpers import client, make_runs  # noqa: E402


@pytest.fixture(scope='module')
def api(tmp_path_factory):
    root = make_runs(tmp_path_factory.mktemp('runs'))
    c, app = client(root)
    with c:
        yield c, root, app


def test_health_and_theme(api):
    c, root, _ = api
    h = c.get('/api/health').json()
    assert h['runs_dir'] == str(root) and h['demo'] is False and 'version' in h
    assert h['torch'] is False
    t = c.get('/api/theme').json()
    assert t['colors']['gem'] == [91, 228, 233] and t['shapes']['player_size'] == [12, 16]


def test_index_embeds_token(api):
    c, _, _ = api
    r = c.get('/')
    assert r.status_code == 200 and 'content="tok"' in r.text
    assert c.get('/static/js/app.js').status_code == 200


def test_runs_list_and_detail(api):
    c, _, _ = api
    runs = {r['name']: r for r in c.get('/api/runs').json()['runs']}
    assert set(runs) == {'legacy', 'grp_s1', 'grp_s2', 'grp_s3'}      # _levels skipped
    leg = runs['legacy']
    assert leg['legacy'] is True and leg['state'] == 'finished' and leg['only_final_policy']
    assert leg['eval']['sampled']['test']['n'] == 64 and leg['eval']['sampled']['test']['k'] == 20
    assert leg['group'].startswith('cfg-') and runs['grp_s1']['group'] == 'grp'
    assert runs['grp_s1']['replays'] == 3 and runs['grp_s1']['sweeps'] == 1
    d = c.get('/api/runs/grp_s1').json()
    assert d['config_full']['group'] == 'grp' and d['files']['replays'] is True
    assert d['sweeps_list'][0]['tag'] == 's1'
    assert c.get('/api/runs/nope').status_code == 404
    assert c.get('/api/runs/_levels').status_code == 404
    assert c.get('/api/runs/..%2F..').status_code == 404


def test_metrics_decimation_since_and_etag(api):
    c, _, _ = api
    r = c.get('/api/runs/legacy/metrics?cols=success,eval_test&max_points=50')
    assert r.status_code == 200
    m = r.json()
    s = m['series']['success']
    assert s['n'] == 120 and len(s['x']) == 50 and s['decimated']
    assert s['x'][0] == 4096 and s['x'][-1] == 120 * 4096          # LTTB keeps the ends
    assert m['series']['eval_test']['n'] == 12                     # sparse column
    etag = r.headers['etag']
    assert c.get('/api/runs/legacy/metrics?cols=success,eval_test&max_points=50',
                 headers={'If-None-Match': etag}).status_code == 304
    inc = c.get('/api/runs/legacy/metrics?cols=success&since=110').json()
    assert inc['series']['success']['x'][0] == 111 * 4096 and inc['series']['success']['n'] == 10
    assert c.get('/api/runs/legacy/metrics?max_points=1').status_code == 422


def test_groups_bands(api):
    c, _, _ = api
    g = {x['group']: x for x in c.get('/api/groups?cols=success').json()['groups']}
    grp = g['grp']
    assert grp['runs'] == ['grp_s1', 'grp_s2', 'grp_s3']
    band = grp['cols']['success']['band']
    assert band and len(band['mean']) == len(grp['cols']['success']['x'])
    assert all(lo <= m <= hi for lo, m, hi in zip(band['lo'], band['mean'], band['hi']))
    leg = next(v for k, v in g.items() if k.startswith('cfg-'))
    assert leg['cols']['success']['band'] is None and 'fewer than 3' in leg['cols']['success']['warning']


def test_replays_index_filters_and_pagination(api):
    c, _, _ = api
    all_ = c.get('/api/replays?limit=500').json()
    assert all_['total'] == 12                     # 9 virtual references + 3
    ids = [x['id'] for x in all_['items']]
    assert '_reference/level_009' in ids and 'grp_s1/0001' in ids
    assert all('actions' not in x for x in all_['items'])
    page1 = c.get('/api/replays?limit=5').json()
    page2 = c.get(f"/api/replays?limit=5&after={page1['next']}").json()
    page3 = c.get(f"/api/replays?limit=5&after={page2['next']}").json()
    assert [x['id'] for x in page1['items'] + page2['items'] + page3['items']] == ids
    assert page3['next'] is None and page1['total'] == 12
    assert c.get('/api/replays?run=grp_s1').json()['total'] == 3
    assert c.get('/api/replays?policy=reference').json()['total'] == 9
    assert c.get('/api/replays?outcome=won&run=_reference').json()['total'] == 9
    assert c.get('/api/replays?level=builtin:2').json()['total'] == 2
    assert c.get('/api/replays?source=final').json()['total'] == 3
    latest = c.get('/api/replays?order=latest&limit=1').json()['items'][0]['id']
    assert latest == 'grp_s1/0003'
    r1 = c.get('/api/replays?order=random&seed=4').json()
    assert r1 == c.get('/api/replays?order=random&seed=4').json()
    assert c.get('/api/replays?limit=501').status_code == 422
    assert c.get('/api/replays?order=bogus').status_code == 422


def test_replay_and_trajectory(api):
    c, root, _ = api
    rec = c.get('/api/replays/grp_s1/2').json()
    assert rec['id'] == 'grp_s1/0002' and rec['actions'] and rec['broken'] is None
    r = c.get('/api/replays/grp_s1/2/trajectory', headers={'Accept-Encoding': 'gzip'})
    assert r.status_code == 200
    t = r.json()                                    # httpx decodes gzip transparently
    n = t['engine_steps']
    assert len(t['xy']) == 2 * (n + 1) and len(t['flags']) == n + 1
    assert t['world'] == [240, 160] and t['drift'] is False and t['result'] == 'won'
    assert {e['type'] for e in t['events']} >= {'gem', 'won'}
    raw = c.get('/api/replays/grp_s1/2/trajectory', headers={'Accept-Encoding': 'identity'})
    assert raw.headers.get('content-encoding') is None and raw.json() == t
    f = c.get('/api/replays/_reference/level_009/trajectory?cam=follow').json()
    assert len(f['cam']) == len(f['xy']) and all(isinstance(v, int) for v in f['cam'])
    assert c.get('/api/replays/grp_s1/99').status_code == 404
    assert c.get('/api/replays/grp_s1/1/trajectory?cam=x').status_code == 422
    assert c.get('/api/levels/7/reference').json()['id'] == '_reference/level_007'


def test_drift_uses_snapshot(api, tmp_path):
    c, root, app = api
    # A replay whose ref regenerates to a different level than its hash.
    rec = c.get('/api/replays/grp_s1/1').json()
    from game import Level
    from game.level import level_hash
    from game.replay_store import store_level
    level4 = Level.from_json('levels/level_004.json')
    digest = store_level(root, level4)
    drifted = {k: v for k, v in rec.items() if k not in ('n', 'id', 'run', 'broken', 'demo')}
    drifted['level'] = dict(rec['level'], ref={'kind': 'builtin', 'n': 1}, hash=digest)
    drifted['actions'] = c.get('/api/levels/4/reference').json()['actions']
    drifted['env'] = {'frame_skip': 1, 'max_steps': 1800, 'physics': None}
    out = append_replays(root / 'grp_s1', [drifted])[0]
    app.state.store.scan()
    t = c.get(f"/api/replays/grp_s1/{out['n']}/trajectory").json()
    assert t['drift'] is True and t['level_source'] == 'snapshot'
    assert t['level_hash'] == digest == level_hash(level4) and t['result'] == 'won'


def test_broken_replays_are_flagged_not_hidden(api):
    c, root, app = api
    with open(root / 'grp_s2' / 'replays.jsonl', 'a') as fh:
        fh.write('{"n": 1, "id": "grp_s2/0001", "actions": [[9, 1]], "level": {"hash": "sha256:x", '
                 '"ref": {"kind": "builtin", "n": 1}}}\n')
        fh.write('this is not json\n')
        fh.write('{"n": 3, "id": "grp_s2/0003", "actions": [[1, 1]], "level": {"hash": '
                 '"sha256:' + '0' * 64 + '", "ref": {"kind": "custom"}}}\n')
    app.state.store.scan()
    items = {x['id']: x for x in c.get('/api/replays?run=grp_s2').json()['items']}
    assert len(items) == 3
    assert items['grp_s2/0001']['broken'].startswith('actions')
    assert any(x['broken'] == 'invalid JSON line' for x in items.values())
    assert 'level store' in items['grp_s2/0003']['broken']
    assert c.get('/api/replays/grp_s2/1/trajectory').status_code in (409, 422)


def test_levels(api):
    c, _, _ = api
    b = c.get('/api/levels/builtin').json()['levels']
    assert [x['n'] for x in b] == list(range(1, 10)) and b[6]['width'] == 480
    ref = json.dumps({'kind': 'pcg', 'preset': 'tiny', 'seed': 1000003})
    lv = c.get('/api/levels/by-ref', params={'ref': ref}).json()
    assert lv['width'] == 240 and 'min_window' in lv['metrics'] and lv['hash'].startswith('sha256:')
    assert 'waypoints' not in lv['source']['metadata']
    assert c.get(f"/api/levels/by-hash/{b[0]['hash']}").json()['name'] == 'Walk'
    assert c.get('/api/levels/by-hash/sha256:' + 'f' * 64).status_code == 404
    assert c.get('/api/levels/by-ref', params={'ref': '{bad'}).status_code == 422
    assert c.get('/api/levels/by-ref', params={'ref': json.dumps({'kind': 'pcg', 'seed': -1, 'preset': 'tiny'})}).status_code == 422
    assert c.get('/api/levels/by-ref', params={'ref': json.dumps({'kind': 'x'})}).status_code == 422
    ex = c.get('/api/levels/expressive?preset=tiny&n=5').json()
    assert len(ex['points']) == 5 and {'leniency', 'linearity', 'seed'} <= set(ex['points'][0])
    assert c.get('/api/levels/expressive?preset=huge&n=500').status_code == 422


def test_sweeps(api):
    c, _, _ = api
    assert c.get('/api/sweeps/grp_s1').json()['sweeps'][0]['tag'] == 's1'
    s = c.get('/api/sweeps/grp_s1/s1/summary').json()
    assert s['n'] == 60
    total = sum(v['k'] for v in s['end_reasons'].values())
    assert total == 60
    assert abs(sum(v['p'] for v in s['end_reasons'].values()) - 1) < 1e-9
    assert s['success']['lo'] <= s['success']['p'] <= s['success']['hi']
    mw = s['difficulty']['min_window']
    assert [b['label'] for b in mw] == ['0', '1', '2', '3'] and all(b['insufficient'] for b in mw)
    h = c.get('/api/sweeps/grp_s1/s1/heatmap?layer=visits').json()
    assert h['cols'] == 30 and h['rows'] == 20 and h['used'] == 60 and h['max'] > 0
    d = c.get('/api/sweeps/grp_s1/s1/heatmap?layer=deaths').json()
    deaths = s['end_reasons']['fall']['k'] + s['end_reasons']['hazard']['k']
    assert sum(d['grids']['fall']) + sum(d['grids']['hazard']) == deaths > 0
    assert c.get('/api/sweeps/grp_s1/nope/summary').status_code == 404
    assert c.get('/api/sweeps/grp_s1/s1/heatmap?layer=x').status_code == 422


def test_trajectory_too_large_is_413(api, monkeypatch):
    c, _, _ = api
    import dashboard.app as appmod
    monkeypatch.setattr(appmod, 'MAX_TRAJECTORY', 100)
    r = c.get('/api/replays/_reference/level_001/trajectory')
    assert r.status_code == 413 and r.json()['error'] == 'trajectory_too_large'


def test_old_runs_are_readable_without_new_files(tmp_path):
    root = tmp_path / 'runs'
    run = root / 'old'
    run.mkdir(parents=True)
    (run / 'progress.csv').write_text('update,env_steps,sps\n1,4096,10\n2,8192,10\n3,122')
    c, _ = client(root)
    with c:
        r = c.get('/api/runs').json()['runs'][0]
        assert r['name'] == 'old' and r['points'] == 2 and r['replays'] == 0
        assert c.get('/api/runs/old/metrics?cols=sps').json()['series']['sps']['n'] == 2
