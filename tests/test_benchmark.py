"""The CLI benchmark must measure real transitions, not terminal no-ops."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_reports_actual_steps_for_each_level():
    result = subprocess.run(
        [sys.executable, '-m', 'tools.benchmark', '--steps', '2000'],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)['levels']
    assert [r['level'] for r in rows] == [f'level_{i:03}' for i in range(1, 10)]
    for row in rows:
        assert row['steps'] == row['effective_steps'] == 2000
        assert row['resets'] > 0
        assert row['seconds'] > 0
        assert row['steps_per_second'] > 0


def test_benchmark_level_7_exceeds_loose_threshold():
    """The benchmark must clear a loose >10k steps/s on the big level."""
    from tools.benchmark import benchmark
    data = benchmark(10_000)
    rows = {r['level']: r for r in data['levels']}
    assert rows['level_007']['steps_per_second'] > 10_000
