"""Stable discrete actions; append future actions without renumbering."""
from enum import IntEnum
import numbers


class Action(IntEnum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5


_DECODE = ((0, False), (-1, False), (1, False),
           (0, True), (-1, True), (1, True))


def decode_action(action: Action | int) -> tuple[int, bool]:
    """Decode to (dx, jump). Accepts any ``numbers.Integral`` that is not a
    bool — which covers numpy integer scalars (``np.int64``/``np.int32``)
    without importing numpy — and rejects bools, floats (even ``2.0``),
    strings, arrays and anything out of range."""
    if isinstance(action, bool) or not isinstance(action, numbers.Integral):
        raise ValueError(f'action: expected Action or int, got {action!r}')
    return _DECODE[Action(int(action))]
