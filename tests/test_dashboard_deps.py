"""Dependency rules of spec 13 (I5, §90): the server never imports torch, numpy,
pygame or rl; game/ and pcg/ stay stdlib-only; rl/ never imports dashboard/."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_server_process_imports_no_heavy_modules(tmp_path):
    pytest.importorskip('fastapi')
    code = (
        "import sys\n"
        "from dashboard.app import create_app\n"
        "from dashboard import demo, jobs, store, stats, analytics, job_schema\n"
        f"app = create_app({str(tmp_path)!r}, watch=False, find_python=False)\n"
        "from fastapi.testclient import TestClient\n"
        "c = TestClient(app, base_url='http://127.0.0.1:8765')\n"
        "assert c.get('/api/replays/_reference/level_007/trajectory?cam=follow').status_code == 200\n"
        "assert c.get('/api/levels/by-ref', params={'ref': '{\"kind\":\"pcg\",\"preset\":\"tiny\",\"seed\":5}'}).status_code == 200\n"
        "bad = [m for m in ('torch', 'numpy', 'pygame', 'rl', 'gymnasium') if m in sys.modules]\n"
        "print(','.join(bad))\n")
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ''


def _imports(path: Path) -> set:
    tree = ast.parse(path.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split('.')[0])
    return mods


@pytest.mark.parametrize('pkg,forbidden', [
    ('game', {'rl', 'rendering', 'dashboard', 'pcg', 'numpy', 'pygame', 'torch', 'gymnasium', 'time'}),
    ('pcg', {'rl', 'rendering', 'dashboard', 'numpy', 'pygame', 'torch', 'gymnasium', 'time'}),
    ('rendering', {'rl', 'dashboard'}),
    ('rl', {'dashboard'}),
    ('dashboard', {'rl', 'torch', 'numpy', 'pygame', 'gymnasium'}),
])
def test_import_graph(pkg, forbidden):
    for path in (ROOT / pkg).rglob('*.py'):
        bad = _imports(path) & forbidden
        assert not bad, f'{path.relative_to(ROOT)} imports {bad}'
