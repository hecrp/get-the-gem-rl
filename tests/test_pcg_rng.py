"""pcg.rng: stable, Python-version-independent PRNG."""
from pcg.rng import SplitMix64, fnv1a, substream


def test_splitmix64_seed_0_golden():
    rng = SplitMix64(0)
    assert rng.next() == 0xe220a8397b1dcdaf
    assert rng.next() == 0x6e789e6aa1b965f4
    assert rng.next() == 0x06c45d188009454f


def test_splitmix64_seed_1_golden():
    rng = SplitMix64(1)
    assert rng.next() == 0x910a2dec89025cc1
    assert rng.next() == 0xbeeb8da1658eec67
    assert rng.next() == 0xf893a2eefb32555e


def test_fnv1a_path_golden():
    assert fnv1a('path#0') == 0x8b2835087cb7476d


def test_seed_is_masked():
    # seeds are reduced mod 2**64, matching the reference
    assert SplitMix64(2 ** 64 + 1).next() == SplitMix64(1).next()


def test_substream_distinct_per_stage_and_attempt():
    streams = [substream(0, a, s) for a in (0, 1) for s in ('layout', 'path', 'hazards', 'decor')]
    firsts = [rng.next() for rng in streams]
    assert len(set(firsts)) == len(firsts)   # every stream is independent


def test_substream_deterministic():
    a = substream(7, 0, 'path')
    b = substream(7, 0, 'path')
    assert [a.next() for _ in range(10)] == [b.next() for _ in range(10)]


def test_below_rint_choice_are_stable():
    assert SplitMix64(42).below(100) == SplitMix64(42).below(100)
    assert SplitMix64(42).rint(3, 9) == SplitMix64(42).rint(3, 9)
    assert SplitMix64(42).choice([10, 20, 30]) == SplitMix64(42).choice([10, 20, 30])