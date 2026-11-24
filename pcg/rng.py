"""Stable, Python-version-independent PRNG for the procedural generator.

``random.Random`` only guarantees the ``random()`` stream across Python
versions, not ``randrange``/``choice``/``shuffle``. A level's identity must be
permanent (benchmarks, train/test splits, reproducing a failure months later),
so the generator uses its own SplitMix64 and never ``random``.
"""
from __future__ import annotations

MASK = (1 << 64) - 1


class SplitMix64:
    """SplitMix64; the sequence is identical on every Python version and machine."""

    def __init__(self, seed: int):
        self.s = seed & MASK

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        return self.next() % n

    def rint(self, a: int, b: int) -> int:
        return a + self.below(b - a + 1)

    def chance(self, p: float) -> bool:
        return self.next() < int(p * (1 << 64))

    def choice(self, seq):
        return seq[self.below(len(seq))]

    def shuffle(self, seq):
        for i in range(len(seq) - 1, 0, -1):
            j = self.below(i + 1)
            seq[i], seq[j] = seq[j], seq[i]

    def weighted(self, items, weights):
        total = sum(weights)
        x = self.below(total)
        for item, w in zip(items, weights):
            if x < w:
                return item
            x -= w
        return items[-1]


def fnv1a(text: str) -> int:
    """Stable string hash; never Python's ``hash()`` (salted per process)."""
    h = 0xCBF29CE484222325
    for b in text.encode():
        h = ((h ^ b) * 0x100000001B3) & MASK
    return h


def substream(seed: int, attempt: int, stage: str) -> SplitMix64:
    """An independent stream per stage: changing one stage never shifts the others."""
    return SplitMix64(SplitMix64(seed ^ fnv1a(f"{stage}#{attempt}")).next())
