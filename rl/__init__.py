"""Reinforcement-learning layer over the engine (Gymnasium).

Nothing in ``game/`` or ``pcg/`` depends on this package.
"""
from .env import GRID_CHANNELS, VEC_FIELDS, GemEnv

__all__ = ['GemEnv', 'GRID_CHANNELS', 'VEC_FIELDS']
