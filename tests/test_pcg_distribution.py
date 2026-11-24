"""LevelDistribution and LevelPack."""
import json
import random

import pytest

from game import Level
from pcg import PRESETS, LevelDistribution, LevelPack, generate_level


def test_get_is_deterministic_and_stable():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=10, num_levels=5)
    a, b = dist.get(2), dist.get(2)
    assert a.tiles == b.tiles
    assert a.metadata['generator']['seed'] == 12


def test_get_maps_index_to_seed():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=100, num_levels=3)
    assert dist.get(0).metadata['generator']['seed'] == 100
    assert dist.get(2).metadata['generator']['seed'] == 102


def test_get_matches_generate_level():
    dist = LevelDistribution(PRESETS['wide'], start_seed=0, num_levels=4)
    for i in range(4):
        assert dist.get(i).tiles == generate_level(i, PRESETS['wide']).tiles


def test_num_levels_bounds():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=0, num_levels=3)
    with pytest.raises(IndexError):
        dist.get(3)
    assert len(dist) == 3


# ---- sample(rng) by duck typing ------------------------------------------

class IntegersOnlyRNG:
    """A double exposing only ``integers`` (numpy.random.Generator's API)."""

    def __init__(self):
        self.calls = []

    def integers(self, low, high=None):
        self.calls.append((low, high))
        if high is None:
            return low // 2
        return (low + high) // 2


def test_sample_accepts_rng_exposing_only_integers():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=7, num_levels=100)
    rng = IntegersOnlyRNG()
    level = dist.sample(rng)
    assert rng.calls == [(0, 100)]
    assert level.metadata['generator']['seed'] == 57  # (0 + 100) // 2 == 50 → seed 7+50
    assert level.metadata['generator']['seed'] == dist.get(50).metadata['generator']['seed']


def test_sample_unbounded_accepts_rng_exposing_only_integers():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=0, num_levels=0)
    rng = IntegersOnlyRNG()
    level = dist.sample(rng)
    assert rng.calls == [(0, 2 ** 63)]
    seed = level.metadata['generator']['seed']
    assert 0 <= seed < 2 ** 63 and seed == 2 ** 62


def test_level_pack_sample_accepts_rng_exposing_only_integers(tmp_path):
    from tools.build_level_pack import build_one
    pack = tmp_path / 'pack.jsonl'
    with pack.open('w', encoding='utf-8') as fh:
        for seed in range(4):
            fh.write(json.dumps(build_one(seed, 'tiny')) + '\n')
    lp = LevelPack(pack)
    rng = IntegersOnlyRNG()
    level = lp.sample(rng)
    assert rng.calls == [(0, 4)]
    assert level.tiles == lp.get(2).tiles


def test_sample_rejects_rng_without_randrange_or_integers():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=0, num_levels=10)
    with pytest.raises(TypeError, match='randrange or integers'):
        dist.sample(object())


def test_sample_accepts_numpy_generator():
    np = pytest.importorskip("numpy")
    dist = LevelDistribution(PRESETS['tiny'], start_seed=0, num_levels=50)
    level = dist.sample(np.random.default_rng(3))
    index = level.metadata['generator']['seed']
    assert 0 <= index < 50
    unbounded = LevelDistribution(PRESETS['tiny'], num_levels=0)
    level = unbounded.sample(np.random.default_rng(3))
    assert 0 <= level.metadata['generator']['seed'] < 2 ** 63


def test_sample_depends_only_on_received_rng():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=5, num_levels=100)
    names_a = [dist.sample(random.Random(42)).name for _ in range(10)]
    names_b = [dist.sample(random.Random(42)).name for _ in range(10)]
    assert names_a == names_b


def test_lru_cache_does_not_change_results():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=0, num_levels=200, cache_size=64)
    first = [dist.get(i).name for i in range(200)]
    second = [dist.get(i).name for i in range(200)]
    assert first == second


def test_unbounded_distribution_samples_and_gets():
    dist = LevelDistribution(PRESETS['tiny'], num_levels=0)
    assert dist.get(999).name == 'gen-v1-999'
    lv = dist.sample(random.Random(0))
    assert isinstance(lv, Level)
    assert len(dist) == 0


def test_level_pack_round_trip(tmp_path):
    from tools.build_level_pack import build_one
    pack = tmp_path / 'pack.jsonl'
    with pack.open('w', encoding='utf-8') as fh:
        for seed in range(6):
            fh.write(json.dumps(build_one(seed, 'tiny')) + '\n')

    lp = LevelPack(pack)
    assert len(lp) == 6
    for i in range(6):
        loaded = lp.get(i)
        assert loaded.tiles == generate_level(i, PRESETS['tiny']).tiles
        assert loaded.metadata['stats']['solver_steps'] >= 1


def test_level_pack_sample_depends_only_on_received_rng(tmp_path):
    from tools.build_level_pack import build_one
    pack = tmp_path / 'pack.jsonl'
    with pack.open('w', encoding='utf-8') as fh:
        for seed in range(4):
            fh.write(json.dumps(build_one(seed, 'tiny')) + '\n')
    lp = LevelPack(pack)
    a = [lp.sample(random.Random(7)).name for _ in range(8)]
    b = [lp.sample(random.Random(7)).name for _ in range(8)]
    assert a == b