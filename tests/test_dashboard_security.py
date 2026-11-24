"""R-14: loopback Host, Origin + token on writes, whitelisted job flags (test 15)."""
import json
import sys

import pytest

pytest.importorskip('fastapi')

from dashboard import job_schema  # noqa: E402
from dashboard_helpers import WRITE_HEADERS, client, make_runs  # noqa: E402


@pytest.fixture(scope='module')
def c(tmp_path_factory):
    root = make_runs(tmp_path_factory.mktemp('runs'))
    cl, app = client(root)
    app.state.jobs.python = sys.executable            # enable validation path
    with cl:
        yield cl


@pytest.mark.parametrize('host', ['evil.com', 'evil.com:8765', '192.168.1.5:8765', '127.0.0.1.evil.com', ''])
def test_non_loopback_host_is_forbidden(c, host):
    r = c.get('/api/health', headers={'Host': host})
    assert r.status_code == 403 and r.json()['error'] == 'forbidden_host'


@pytest.mark.parametrize('host', ['127.0.0.1:8765', 'localhost:8765', '[::1]:8765', 'localhost'])
def test_loopback_hosts_are_allowed(c, host):
    assert c.get('/api/health', headers={'Host': host}).status_code == 200


def test_writes_need_origin_and_token(c):
    body = {'kind': 'rollout', 'args': {'run': 'grp_s1'}}
    assert c.post('/api/jobs', json=body).json()['error'] == 'forbidden_origin'
    assert c.post('/api/jobs', json=body, headers={'Origin': 'http://evil.com'}).status_code == 403
    r = c.post('/api/jobs', json=body, headers={'Origin': 'http://127.0.0.1:8765'})
    assert r.status_code == 403 and r.json()['error'] == 'forbidden_token'
    r = c.post('/api/jobs', json=body, headers={**WRITE_HEADERS, 'X-Dashboard-Token': 'wrong'})
    assert r.status_code == 403
    assert c.delete('/api/jobs/1').status_code == 403
    assert c.delete('/api/jobs/1', headers=WRITE_HEADERS).status_code == 404   # passes security


@pytest.mark.parametrize('kind,args', [
    ('train', {'shell': 'true'}),
    ('train', {'name': 'x; rm -rf /'}),
    ('train', {'name': '../../etc'}),
    ('train', {'name': '_reserved'}),
    ('train', {'levels': '1-6; rm -rf ~'}),
    ('train', {'preset': 'tiny && curl evil'}),
    ('train', {'total_steps': 200_000_001}),
    ('train', {'total_steps': '1000'}),
    ('train', {'lr': 1e9}),
    ('train', {'no_goal_vector': 'yes'}),
    ('train', {'device': 'cuda:0; ls'}),
    ('train', {'resume': '../../model.pt'}),
    ('rollout', {'run': 'grp_s1', 'ckpt': '/etc/passwd'}),
    ('rollout', {'run': 'grp_s1', 'tag': '$(whoami)'}),
    ('rollout', {'run': 'grp_s1', 'max_episodes': 10}),          # sweep-only flag
    ('sweep', {'run': 'grp_s1', 'max_episodes': 20_001}),
    ('nuke', {}),
])
def test_invalid_jobs_are_422(c, kind, args):
    r = c.post('/api/jobs', json={'kind': kind, 'args': args}, headers=WRITE_HEADERS)
    assert r.status_code == 422, r.text


def test_non_finite_numbers_rejected(c):
    for bad in ('NaN', 'Infinity', '-Infinity'):
        r = c.post('/api/jobs', content='{"kind": "train", "args": {"lr": %s}}' % bad,
                   headers={**WRITE_HEADERS, 'Content-Type': 'application/json'})
        assert r.status_code == 422, bad


def test_body_limits_and_shape(c):
    big = {'kind': 'train', 'args': {'name': 'x' * 70_000}}
    r = c.post('/api/jobs', content=json.dumps(big), headers={**WRITE_HEADERS, 'Content-Type': 'application/json'})
    assert r.status_code == 413
    r = c.post('/api/jobs', content='{not json', headers={**WRITE_HEADERS, 'Content-Type': 'application/json'})
    assert r.status_code == 422
    r = c.post('/api/jobs', json={'kind': 'train', 'args': {}, 'extra': 1}, headers=WRITE_HEADERS)
    assert r.status_code == 422


def test_existing_run_is_409_with_suggestion(c):
    r = c.post('/api/jobs', json={'kind': 'train', 'args': {'name': 'legacy'}}, headers=WRITE_HEADERS)
    assert r.status_code == 409 and 'legacy_2' in r.json()['detail']


def test_argv_is_a_list_never_a_shell_string():
    args = job_schema.validate('train', {'name': 'ok', 'preset': 'tiny', 'lr': 0.0005,
                                         'no_goal_vector': True, 'resume': 'latest'})
    argv = job_schema.argv('train', args, '/py', '/runs')
    assert argv[:3] == ['/py', '-m', 'rl.ppo'] and '--no-goal-vector' in argv
    assert argv[argv.index('--lr') + 1] == '0.0005'
    assert argv[argv.index('--resume') + 1] == '--runs-dir'       # bare --resume = latest
    argv = job_schema.argv('sweep', job_schema.validate('sweep', {'run': 'r', 'policy': 'sampled'}), '/py', '/runs')
    assert argv[3:5] == ['--mode', 'sweep'] and '--sampled' in argv


def test_jobs_disabled_without_torch(tmp_path):
    root = make_runs(tmp_path / 'r')
    cl, app = client(root)
    with cl:
        r = cl.post('/api/jobs', json={'kind': 'train', 'args': {}}, headers=WRITE_HEADERS)
        assert r.status_code == 503 and r.json()['error'] == 'no_torch'
        assert cl.get('/api/jobs').json()['enabled'] is False
