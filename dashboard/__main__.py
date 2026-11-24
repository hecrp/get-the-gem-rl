"""``python -m dashboard`` — serve the dashboard on http://127.0.0.1:8765."""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '::1')


def main(argv=None) -> int:
    if sys.version_info < (3, 11):
        print('dashboard: Python >= 3.11 is required', file=sys.stderr)
        return 2
    ap = argparse.ArgumentParser(prog='python -m dashboard', description=__doc__)
    ap.add_argument('--runs-dir', default=str(ROOT / 'runs'))
    ap.add_argument('--host', default='127.0.0.1', help='loopback only')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--demo', action='store_true',
                    help='serve synthetic demo data from <runs-dir>/_demo (no PyTorch needed)')
    ap.add_argument('--train-python', default=None,
                    help='interpreter for training/recording jobs (default: one with torch)')
    ap.add_argument('--no-jobs', action='store_true', help='disable training/recording jobs')
    args = ap.parse_args(argv)
    if args.host not in LOOPBACK_HOSTS:
        print(f'dashboard: refusing to listen on {args.host!r}: loopback only', file=sys.stderr)
        return 2
    try:
        import uvicorn
    except ImportError:
        print('dashboard: install requirements-dashboard.txt (fastapi, uvicorn)', file=sys.stderr)
        return 2
    from .app import create_app
    runs_dir = Path(args.runs_dir)
    if args.demo:
        from . import demo
        print('building demo data ...', flush=True)
        runs_dir = demo.build(runs_dir / '_demo')
    token = secrets.token_urlsafe(24)
    app = create_app(runs_dir, demo=args.demo, train_python=args.train_python, token=token,
                     find_python=not args.no_jobs)
    if args.no_jobs:
        app.state.jobs.python = None
    host = '[::1]' if args.host == '::1' else args.host
    print(f'Get the Gem dashboard: http://{host}:{args.port}/  (runs: {runs_dir})', flush=True)
    print(f'token: {token}', flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
    return 0


if __name__ == '__main__':
    sys.exit(main())
