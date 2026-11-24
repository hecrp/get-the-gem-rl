"""Procedural level generator.

``generate_level(seed, config)`` deterministically produces a winnable
``game.Level``. ``LevelDistribution`` exposes Procgen-style level sets for
train/test splits by seed range.
"""
from __future__ import annotations

import json
from collections import OrderedDict

from game import Level

from .generator import (
    GENERATOR_VERSION,
    GenConfig,
    GenFail,
    GenerationError,
    Ledge,
    _config_dict,
    generate_level,
    level_to_dict,
)
from .jump_table import JumpTable, build_jump_table
from .metrics import LevelMetrics, metrics

__all__ = [
    'generate_level', 'GenConfig', 'GenFail', 'GenerationError',
    'GENERATOR_VERSION', 'LevelDistribution', 'LevelPack', 'PRESETS',
    'JumpTable', 'build_jump_table', 'LevelMetrics', 'metrics', 'level_to_dict',
    'pcg_ref', 'preset_name',
]

PRESETS: dict[str, GenConfig] = {
    'tiny': GenConfig(cols=30, rows=20, difficulty=0.2),
    'small': GenConfig(cols=60, rows=40),
    'wide': GenConfig(cols=90, rows=20),
    'tall': GenConfig(cols=30, rows=60),
    'large': GenConfig(cols=120, rows=80),
    'huge': GenConfig(cols=240, rows=160, difficulty=0.7),
}


def preset_name(config: GenConfig) -> str | None:
    """The name of the preset whose config equals ``config``, else ``None``."""
    for name, preset in PRESETS.items():
        if preset == config:
            return name
    return None


def pcg_ref(seed: int, config: GenConfig, preset: str | None = None) -> dict:
    """JSON-native provenance of a generated level (``LevelRef``):
    the full config is kept because presets may change over time."""
    return {'kind': 'pcg', 'preset': preset if preset is not None else preset_name(config),
            'seed': seed, 'config': _config_dict(config),
            'generator_version': GENERATOR_VERSION}


def _draw(rng, bound: int | None) -> int:
    """Draw an index below ``bound`` (``None`` = unbounded 63-bit seed) from
    the caller's RNG by duck typing, without importing numpy: an object with
    ``randrange``/``getrandbits`` (``random.Random``) or with ``integers``
    (``numpy.random.Generator``)."""
    if hasattr(rng, 'randrange'):
        return rng.randrange(bound) if bound is not None else rng.getrandbits(63)
    if hasattr(rng, 'integers'):
        return int(rng.integers(0, 2 ** 63 if bound is None else bound))
    raise TypeError(f'sample: rng must expose randrange or integers, got {rng!r}')


class LevelDistribution:
    """A Procgen-style level set over seeds ``[start_seed, start_seed + num_levels)``.

    ``num_levels = 0`` means unbounded (63-bit seeds). ``sample(rng)`` draws
    only from the caller's RNG; ``get(index)`` is deterministic. Levels are
    cached per seed with a small LRU so that, for finite sets, generation cost
    disappears after warm-up. It knows nothing of episodes or rewards: it is a
    pure source of levels.
    """

    def __init__(self, config: GenConfig, start_seed: int = 0, num_levels: int = 0,
                 cache_size: int = 256, *, preset: str | None = None):
        if type(start_seed) is not int or start_seed < 0:
            raise ValueError('start_seed: expected a non-negative integer')
        if type(num_levels) is not int or num_levels < 0:
            raise ValueError('num_levels: expected a non-negative integer')
        self.config = config
        self.start_seed = start_seed
        self.num_levels = num_levels
        self._cache_size = cache_size
        self._cache: OrderedDict[int, object] = OrderedDict()
        self.preset = preset if preset is not None else preset_name(config)
        #: ``LevelRef`` of the level last returned by ``get``/``sample`` (or
        #: ``None``). Bookkeeping only: it never touches an RNG.
        self.last_ref: dict | None = None

    def __len__(self) -> int:
        return self.num_levels

    def get(self, index: int):
        if self.num_levels and not (0 <= index < self.num_levels):
            raise IndexError(f'index {index} out of range [0, {self.num_levels})')
        seed = self.start_seed + index
        self.last_ref = pcg_ref(seed, self.config, self.preset)
        level = self._cache.get(seed)
        if level is not None:
            self._cache.move_to_end(seed)
            return level
        level = generate_level(seed, self.config)
        self._cache[seed] = level
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return level

    def sample(self, rng):
        if self.num_levels:
            return self.get(_draw(rng, self.num_levels))
        seed = _draw(rng, None)
        self.last_ref = pcg_ref(seed, self.config, self.preset)
        return generate_level(seed, self.config)


class LevelPack:
    """A frozen JSON Lines pack (``build_level_pack``), same interface as
    ``LevelDistribution``: ``sample``, ``get``, ``len``. Packs freeze a
    benchmark against ``GENERATOR_VERSION`` changes."""

    def __init__(self, path):
        self.path = str(path)
        self._levels = None
        self._refs: dict[int, dict] = {}
        #: ``LevelRef`` of the level last returned by ``get``/``sample``:
        #: ``{"kind": "custom", "hash", "pack", "index"}`` (packs freeze levels,
        #: so they are addressed by content, not by generator seed).
        self.last_ref: dict | None = None

    def _ref(self, index: int) -> dict:
        ref = self._refs.get(index)
        if ref is None:
            from pathlib import Path

            from game.level import level_hash
            ref = {'kind': 'custom', 'hash': level_hash(self._levels[index]),
                   'pack': Path(self.path).name, 'index': index}
            self._refs[index] = ref
        return dict(ref)

    def _load(self):
        levels = []
        with open(self.path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    levels.append(Level.from_dict(json.loads(line)))
        self._levels = levels

    def __len__(self):
        if self._levels is None:
            self._load()
        return len(self._levels)

    def get(self, index: int):
        if self._levels is None:
            self._load()
        level = self._levels[index]
        self.last_ref = self._ref(index % len(self._levels))
        return level

    def sample(self, rng):
        if self._levels is None:
            self._load()
        index = _draw(rng, len(self._levels))
        self.last_ref = self._ref(index)
        return self._levels[index]
