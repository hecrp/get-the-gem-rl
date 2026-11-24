"""Immutable geometry, tiles parsing and validated JSON loading (no rendering imports)."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

from .collision import overlaps
from .constants import (EXIT_H, EXIT_W, GEM_SIZE, GRID_CELL, MAX_WORLD_H,
                        MAX_WORLD_W, PLAYER_H, PLAYER_W, TILE_SIZE)
from .entities import Exit, Gem, Hazard, Platform, Rect


# Distinguish an omitted condition from an explicitly invalid None.
_DEFAULT_WIN_CONDITION = object()
_SUPPORTED_VERSIONS = (1, 2, 3)


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: expected a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label}: expected a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label}: expected a finite number")
    return result


def _sequence(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label}: expected {length} coordinates")
    return tuple(_number(v, f"{label}[{i}]") for i, v in enumerate(value))


def _positive_integer(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}: expected a positive integer")
    return value


def _rect(value, kind, label):
    if isinstance(value, Rect):
        value = (value.x, value.y, value.w, value.h)
    values = _sequence(value, 4, label)
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"{label}: width and height must be positive")
    return kind(*values)


def _merge_solids(solids):
    """Fuse `#` tiles: consecutive columns per row into spans, then merge a
    span into the rectangle directly above it when x/width match."""
    rects = []          # each is [x, y, w, h]
    above = {}          # (x, w) -> index of the rectangle directly above
    for r in sorted(solids):
        cols = sorted(solids[r])
        spans = []
        start = prev = cols[0]
        for c in cols[1:]:
            if c == prev + 1:
                prev = c
            else:
                spans.append((start, prev))
                start = prev = c
        spans.append((start, prev))
        new_above = {}
        for c0, c1 in spans:
            x = TILE_SIZE * c0
            w = TILE_SIZE * (c1 - c0 + 1)
            key = (x, w)
            if key in above:
                rects[above[key]][3] += TILE_SIZE
                new_above[key] = above[key]
            else:
                new_above[key] = len(rects)
                rects.append([x, TILE_SIZE * r, w, TILE_SIZE])
        above = new_above
    return rects


def _merge_spikes(spikes):
    """Fuse `^` tiles horizontally into [x, r*8+4, 8*count, 4] hazards."""
    hazards = []
    for r in sorted(spikes):
        cols = sorted(spikes[r])
        spans = []
        start = prev = cols[0]
        for c in cols[1:]:
            if c == prev + 1:
                prev = c
            else:
                spans.append((start, prev))
                start = prev = c
        spans.append((start, prev))
        for c0, c1 in spans:
            hazards.append([TILE_SIZE * c0, TILE_SIZE * r + 4,
                            TILE_SIZE * (c1 - c0 + 1), 4])
    return hazards


def _parse_tiles(tiles):
    """Translate tile rows into geometry + entity positions."""
    rows = [str(row) for row in tiles]
    if not rows:
        raise ValueError('tiles: expected at least one row')
    width = len(rows[0])
    if width == 0:
        raise ValueError('tiles: expected non-empty rows')
    spawn = gem = exit_pos = None
    solids = defaultdict(list)   # row -> list of columns
    spikes = defaultdict(list)
    for r, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(
                f'tiles: row {r} has length {len(row)} but the first row has {width}')
        for c, ch in enumerate(row):
            if ch == '.':
                continue
            if ch == '#':
                solids[r].append(c)
            elif ch == '^':
                spikes[r].append(c)
            elif ch == 'P':
                if spawn is not None:
                    raise ValueError('tiles: multiple P spawn markers')
                spawn = (TILE_SIZE * c, TILE_SIZE * (r + 1) - PLAYER_H)
            elif ch == 'G':
                if gem is not None:
                    raise ValueError('tiles: multiple G gem markers')
                gem = (TILE_SIZE * c, TILE_SIZE * r)
            elif ch == 'D':
                if exit_pos is not None:
                    raise ValueError('tiles: multiple D door markers')
                exit_pos = (TILE_SIZE * c, TILE_SIZE * (r + 1) - EXIT_H)
            else:
                raise ValueError(f'tiles: unknown character {ch!r} at row {r}, column {c}')
    return (width * TILE_SIZE, len(rows) * TILE_SIZE,
            spawn, gem, exit_pos,
            _merge_solids(solids), _merge_spikes(spikes))


def _build_grid(width, height, solids, hazards):
    """Broad-phase uniform grid; each cell holds ascending platform/hazard indices."""
    gw = (width + GRID_CELL - 1) // GRID_CELL
    gh = (height + GRID_CELL - 1) // GRID_CELL

    def fill(rects):
        cells = [[] for _ in range(gw * gh)]
        for i, r in enumerate(rects):
            c0 = int(math.floor(r.x / GRID_CELL))
            c1 = int(math.ceil((r.x + r.w) / GRID_CELL)) - 1
            r0 = int(math.floor(r.y / GRID_CELL))
            r1 = int(math.ceil((r.y + r.h) / GRID_CELL)) - 1
            for ry in range(max(0, r0), min(gh, r1 + 1)):
                for cx in range(max(0, c0), min(gw, c1 + 1)):
                    cells[ry * gw + cx].append(i)
        return tuple(tuple(c) for c in cells)

    return gw, gh, fill(solids), fill(hazards)


@dataclass(frozen=True, slots=True)
class Level:
    width: int
    height: int
    player_spawn: tuple[float, float]
    gem: Gem
    platforms: tuple[Platform, ...]
    hazards: tuple[Hazard, ...] = ()
    exit: Exit | None = None
    version: int = 1
    name: str = ""
    _metadata: dict = field(default_factory=dict, repr=False, hash=False)
    _extras: dict = field(default_factory=dict, repr=False, hash=False)
    max_steps: int | None = None
    tiles: tuple[str, ...] | None = field(default=None, compare=False, repr=False)
    _grid: tuple = field(default=(), compare=False, repr=False, hash=False)
    win_condition: str = field(default=_DEFAULT_WIN_CONDITION)

    def __post_init__(self):
        if self.win_condition is _DEFAULT_WIN_CONDITION:
            object.__setattr__(self, 'win_condition',
                               'gem' if self.exit is None else 'exit_with_gem')
        if (not isinstance(self.win_condition, str) or
                self.win_condition not in ('gem', 'exit_with_gem')):
            raise ValueError('win_condition: expected gem or exit_with_gem')
        if self.win_condition == 'exit_with_gem' and self.exit is None:
            raise ValueError('win_condition: exit_with_gem requires exit')
        _positive_integer(self.width, 'width')
        _positive_integer(self.height, 'height')
        self._validate_world_size()
        if type(self.version) is not int:
            raise ValueError('version: expected an integer')
        if self.version not in _SUPPORTED_VERSIONS:
            raise ValueError('version: expected 1, 2 or 3')
        if not isinstance(self.name, str):
            raise ValueError('name: expected a string')
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 1):
            raise ValueError('max_steps: expected None or a positive integer')
        if self.tiles is not None:
            object.__setattr__(self, 'tiles', tuple(str(r) for r in self.tiles))
        for key in ('metadata', 'extras'):
            value = getattr(self, '_' + key)
            if not isinstance(value, dict):
                raise ValueError(f'{key}: expected an object')
            object.__setattr__(self, '_' + key, deepcopy(value))
        spawn = _sequence(self.player_spawn, 2, 'player_spawn')
        object.__setattr__(self, 'player_spawn', spawn)
        for key, kind in (('platforms', Platform), ('hazards', Hazard)):
            values = getattr(self, key)
            if not isinstance(values, (list, tuple)):
                raise ValueError(f'{key}: expected a list of rectangles')
            object.__setattr__(self, key, tuple(
                _rect(v, kind, f'{key}[{i}]') for i, v in enumerate(values)))
        object.__setattr__(self, 'gem', _rect(self.gem, Gem, 'gem'))
        if (self.gem.w, self.gem.h) != (GEM_SIZE, GEM_SIZE):
            raise ValueError('gem: incorrect hitbox size')
        if self.exit is not None:
            object.__setattr__(self, 'exit', _rect(self.exit, Exit, 'exit'))
            if (self.exit.w, self.exit.h) != (EXIT_W, EXIT_H):
                raise ValueError('exit: incorrect hitbox size')
            e = self.exit
            if not (0 <= e.x and 0 <= e.y and
                    e.x + e.w <= self.width and e.y + e.h <= self.height):
                raise ValueError('exit: outside world bounds')
            if any(overlaps(e, p) for p in self.platforms):
                raise ValueError('exit: overlaps platforms')
            if overlaps(e, self.gem):
                raise ValueError('exit: overlaps gem')
            if any(overlaps(e, h) for h in self.hazards):
                raise ValueError('exit: overlaps hazards')
        player = Rect(*spawn, PLAYER_W, PLAYER_H)
        for label, rect in (('player_spawn', player), ('gem', self.gem)):
            if not (0 <= rect.x and 0 <= rect.y and
                    rect.x + rect.w <= self.width and rect.y + rect.h <= self.height):
                raise ValueError(f'{label}: hitbox outside world bounds')
        if any(overlaps(player, p) for p in self.platforms):
            raise ValueError('player_spawn: overlaps platforms')
        if any(overlaps(player, h) for h in self.hazards):
            raise ValueError('player_spawn: overlaps hazards')
        if overlaps(player, self.gem):
            raise ValueError('player_spawn: overlaps gem')
        if any(p.x <= self.gem.x and p.y <= self.gem.y and
               p.x + p.w >= self.gem.x + self.gem.w and
               p.y + p.h >= self.gem.y + self.gem.h for p in self.platforms):
            raise ValueError('gem: fully contained in a platform')
        object.__setattr__(self, '_grid',
                           _build_grid(self.width, self.height,
                                       self.platforms, self.hazards))

    def _validate_world_size(self):
        if self.width % TILE_SIZE or self.height % TILE_SIZE:
            raise ValueError('size: width and height must be multiples of 8')
        if self.width < 240 or self.height < 160:
            raise ValueError('size: minimum world size is 240x160')
        if self.width > MAX_WORLD_W or self.height > MAX_WORLD_H:
            raise ValueError('size: maximum world size exceeded')

    @property
    def metadata(self) -> dict:
        """A defensive deep copy; callers cannot mutate shared level metadata."""
        return deepcopy(self._metadata)

    @property
    def extras(self) -> dict:
        """Unknown fields preserved for future loaders, returned defensively."""
        return deepcopy(self._extras)

    def _indices_near(self, x, y, w, h, slot):
        gw, gh, solids, hazards = self._grid
        cells = solids if slot == 2 else hazards
        # A one-cell margin makes this a superset of every rectangle that could
        # overlap during axis resolution (snap-back is bounded by one cell).
        c0 = int(math.floor(x / GRID_CELL)) - 1
        c1 = int(math.ceil((x + w) / GRID_CELL)) - 1 + 1
        r0 = int(math.floor(y / GRID_CELL)) - 1
        r1 = int(math.ceil((y + h) / GRID_CELL)) - 1 + 1
        seen = set()
        for ry in range(max(0, r0), min(gh, r1 + 1)):
            for cx in range(max(0, c0), min(gw, c1 + 1)):
                seen.update(cells[ry * gw + cx])
        return tuple(sorted(seen))

    def solids_near(self, x, y, w, h) -> tuple[int, ...]:
        """Ascending, deduplicated indices of platforms near (a margin around) the AABB."""
        return self._indices_near(x, y, w, h, 2)

    def hazards_near(self, x, y, w, h) -> tuple[int, ...]:
        """Ascending, deduplicated indices of hazards near (a margin around) the AABB."""
        return self._indices_near(x, y, w, h, 3)

    def to_dict(self) -> dict:
        """JSON-native dict such that ``Level.from_dict(level.to_dict())`` is
        an equal level: tiles levels keep their tiles (the same dict as
        ``pcg.level_to_dict`` for generated levels), others use rectangles
        ``[x, y, w, h]`` with gem/exit as ``[x, y]``. See ``level_hash``."""
        return _level_to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Level':
        try:
            if not isinstance(data, dict):
                raise ValueError('root: expected an object')
            known = {'width', 'height', 'player_spawn', 'gem', 'platforms',
                     'hazards', 'exit', 'version', 'name', 'metadata',
                     'win_condition', 'tiles', 'tile_size', 'max_steps'}
            tiles = data.get('tiles')
            tile_size = data.get('tile_size')
            if tile_size is not None and tile_size != TILE_SIZE:
                raise ValueError('tile_size: only 8 is supported')

            if tiles is not None:
                width, height, t_spawn, t_gem, t_exit, tile_platforms, tile_hazards = \
                    _parse_tiles(tiles)
                if 'width' in data and data['width'] != width:
                    raise ValueError('width: must match tiles columns x 8')
                if 'height' in data and data['height'] != height:
                    raise ValueError('height: must match tiles rows x 8')
                if t_spawn is None:
                    if 'player_spawn' not in data:
                        raise ValueError('player_spawn: required when tiles has no P')
                    spawn = data['player_spawn']
                else:
                    if 'player_spawn' in data:
                        raise ValueError('player_spawn: cannot be in both tiles and explicit')
                    spawn = t_spawn
                if t_gem is None:
                    if 'gem' not in data:
                        raise ValueError('gem: required when tiles has no G')
                    gem = data['gem']
                else:
                    if 'gem' in data:
                        raise ValueError('gem: cannot be in both tiles and explicit')
                    gem = t_gem
                if t_exit is not None:
                    if data.get('exit') is not None:
                        raise ValueError('exit: cannot be in both tiles and explicit')
                    exit_pos = t_exit
                else:
                    exit_pos = data.get('exit')
                platforms = tile_platforms + list(data.get('platforms', []))
                hazards = tile_hazards + list(data.get('hazards', []))
            else:
                width = _positive_integer(data['width'], 'width')
                height = _positive_integer(data['height'], 'height')
                spawn = data['player_spawn']
                gem = data['gem']
                exit_pos = data.get('exit')
                platforms = data['platforms']
                hazards = data.get('hazards', ())

            gem_rect = Gem(*_sequence(gem, 2, 'gem'), GEM_SIZE, GEM_SIZE)
            exit_rect = (None if exit_pos is None else
                         Exit(*_sequence(exit_pos, 2, 'exit'), EXIT_W, EXIT_H))
            tiles_tuple = tuple(str(r) for r in tiles) if tiles is not None else None
            extras = {k: v for k, v in data.items() if k not in known}
            return cls(width=width, height=height,
                       player_spawn=spawn, gem=gem_rect,
                       platforms=platforms, hazards=hazards, exit=exit_rect,
                       version=data.get('version', 1),
                       name=data.get('name', ''),
                       _metadata=data.get('metadata', {}),
                       _extras=extras,
                       max_steps=data.get('max_steps'),
                       tiles=tiles_tuple,
                       win_condition=data.get('win_condition', _DEFAULT_WIN_CONDITION))
        except KeyError as exc:
            raise ValueError(f'<dict>: {exc.args[0]}: required field missing') from exc
        except ValueError as exc:
            raise ValueError(f'<dict>: {exc}') from exc

    @classmethod
    def from_json(cls, path: str | Path) -> 'Level':
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f'{path}: JSON: {exc}') from exc
        try:
            return cls.from_dict(data)
        except ValueError as exc:
            raise ValueError(f'{path}: {exc}') from exc

# -- canonical serialisation ----------------------------------

def canonical_json(obj) -> str:
    """The one canonical JSON text used for hashing levels: sorted keys, no
    insignificant whitespace, UTF-8 (non-ASCII kept as is)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _num(v: float):
    """Integral floats are written as ints (``16`` rather than ``16.0``) so a
    round trip through ``from_dict`` (which stores floats) is exact and the
    canonical text matches the hand-written JSON files."""
    return int(v) if float(v).is_integer() else v


def _rect_list(r) -> list:
    return [_num(r.x), _num(r.y), _num(r.w), _num(r.h)]


def _level_to_dict(level: 'Level') -> dict:
    if level.tiles is not None:
        # Tiles carry the geometry; anything given explicitly on top of the
        # tiles (``from_dict`` appends it after the tile-derived rectangles,
        # and spawn/gem/exit when the tiles lack P/G/D) is kept explicitly.
        (_, _, t_spawn, t_gem, t_exit, t_platforms, t_hazards) = _parse_tiles(level.tiles)
        out = {'version': level.version, 'name': level.name, 'tile_size': TILE_SIZE,
               'win_condition': level.win_condition, 'max_steps': level.max_steps,
               'tiles': list(level.tiles), 'metadata': level.metadata}
        extra_platforms = level.platforms[len(t_platforms):]
        extra_hazards = level.hazards[len(t_hazards):]
        if extra_platforms:
            out['platforms'] = [_rect_list(p) for p in extra_platforms]
        if extra_hazards:
            out['hazards'] = [_rect_list(h) for h in extra_hazards]
        if t_spawn is None:
            out['player_spawn'] = [_num(v) for v in level.player_spawn]
        if t_gem is None:
            out['gem'] = [_num(level.gem.x), _num(level.gem.y)]
        if t_exit is None and level.exit is not None:
            out['exit'] = [_num(level.exit.x), _num(level.exit.y)]
    else:
        out = {'version': level.version, 'name': level.name,
               'width': level.width, 'height': level.height,
               'player_spawn': [_num(v) for v in level.player_spawn],
               'gem': [_num(level.gem.x), _num(level.gem.y)],
               'platforms': [_rect_list(p) for p in level.platforms],
               'hazards': [_rect_list(h) for h in level.hazards],
               'win_condition': level.win_condition,
               'max_steps': level.max_steps,
               'metadata': level.metadata}
        if level.exit is not None:
            out['exit'] = [_num(level.exit.x), _num(level.exit.y)]
    for key, value in level.extras.items():
        out.setdefault(key, value)
    return out


def level_hash(level: 'Level') -> str:
    """Content address of a level: ``"sha256:" + hexdigest`` of its canonical JSON."""
    import hashlib
    return 'sha256:' + hashlib.sha256(
        canonical_json(level.to_dict()).encode('utf-8')).hexdigest()
