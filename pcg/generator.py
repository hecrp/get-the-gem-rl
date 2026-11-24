"""Procedural level generator: the critical path first, hazards and distractors after.

Only the jump table actually simulates (with the real ``game.Game``); the
generator itself just reasons over measured reachability. Each stage draws
from its own substream so that editing one stage never shifts the others, and
the whole pipeline uses integer arithmetic only, so ``(version, seed, config)`` identifies a level
permanently.

The pipeline:

    1 layout (room tree + anchors) -> 2 critical path (reversible ledges)
    -> 3 floor + hazards -> 4 distractors -> 5 emit a tiles-v3 ``game.Level``

A failed attempt raises ``GenFail`` and is retried with ``attempt + 1``;
after ``max_attempts`` it raises ``GenerationError``.
"""
from __future__ import annotations

from dataclasses import dataclass

from game import Level
from game.constants import TILE_SIZE
from game.physics import PhysicsConfig

from .jump_table import JumpTable, build_jump_table
from .rng import substream

GENERATOR_VERSION = 1

EMPTY, SOLID, SPIKE = 0, 1, 2

ROOM_C, ROOM_R = 30, 20          # one-screen rooms

# Jump-table search bounds for a reversible move.
G_MAX, D_UP, D_DOWN = 6, 5, 8


class GenFail(Exception):
    """Internal: a single attempt could not build a valid level."""


class GenerationError(Exception):
    """Public: no valid level after ``max_attempts``, or verification failed."""


@dataclass(frozen=True, slots=True)
class GenConfig:
    cols: int = 60                 # multiple of 30
    rows: int = 40                 # multiple of 20
    difficulty: float = 0.5        # 0..1
    floor_pits: bool = True
    ledge_spikes: bool = True
    distractors: bool = True
    win_condition: str = "exit_with_gem"   # or "gem": no door, no return trip
    max_attempts: int = 20
    verify: bool = False           # force guided-solver verification (offline)

    def __post_init__(self):
        if self.cols % ROOM_C or self.cols < ROOM_C:
            raise ValueError('cols: must be a positive multiple of 30')
        if self.rows % ROOM_R or self.rows < ROOM_R:
            raise ValueError('rows: must be a positive multiple of 20')
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError('difficulty: must be within [0, 1]')
        if self.win_condition not in ('exit_with_gem', 'gem'):
            raise ValueError('win_condition: expected "exit_with_gem" or "gem"')
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError('max_attempts: expected a positive integer')


@dataclass
class Ledge:
    r: int
    c0: int
    c1: int

    @property
    def mid(self) -> int:
        return (self.c0 + self.c1) // 2


def _config_dict(cfg: GenConfig) -> dict:
    """The generation-relevant config, in a stable key order (verify is a
    one-shot request, not part of a level's identity, so it is excluded)."""
    return {
        'cols': cfg.cols, 'rows': cfg.rows, 'difficulty': cfg.difficulty,
        'floor_pits': cfg.floor_pits, 'ledge_spikes': cfg.ledge_spikes,
        'distractors': cfg.distractors, 'win_condition': cfg.win_condition,
        'max_attempts': cfg.max_attempts,
    }


def generate_level(seed: int, config: GenConfig = GenConfig(),
                   physics: PhysicsConfig | None = None) -> Level:
    table = build_jump_table(physics)
    for attempt in range(config.max_attempts):
        try:
            level = _generate_once(seed, attempt, config, table)
            break
        except GenFail:
            continue
    else:
        raise GenerationError(
            f'seed {seed}: no valid level after {config.max_attempts} attempts')
    if config.verify:
        from tools.solve_level import solve
        sequence, _ = solve(level, k=3)
        if sequence is None:
            raise GenerationError(f'seed {seed}: verification failed (unsolvable)')
    return level


def _generate_once(seed: int, attempt: int, cfg: GenConfig, table: JumpTable) -> Level:
    C, R, D = cfg.cols, cfg.rows, cfg.difficulty
    window = table.window
    grid = [bytearray(C) for _ in range(R)]
    reserved = [bytearray(C) for _ in range(R)]
    FLOOR = R - 2
    for c in range(C):
        grid[FLOOR][c] = grid[FLOOR + 1][c] = SOLID

    # ---- stage 1: macro-path over the room grid (Spelunky-style) ----------
    rng = substream(seed, attempt, "layout")
    RC, RR = max(C // ROOM_C, 1), max(R // ROOM_R, 1)
    start_room = (rng.below(RC), RR - 1)
    order, stack, seen, parent = [], [start_room], {start_room}, {start_room: None}
    while stack:                                           # random DFS => spanning tree
        cur = stack[-1]
        nb = [(cur[0] + dx, cur[1] + dy) for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1))]
        nb = [n for n in nb if 0 <= n[0] < RC and 0 <= n[1] < RR and n not in seen]
        if not nb:
            stack.pop()
            continue
        n = rng.choice(nb)
        seen.add(n)
        parent[n] = cur
        stack.append(n)
        order.append(n)

    def depth(n):
        k = 0
        while parent[n] is not None:
            n = parent[n]
            k += 1
        return k

    rooms = [start_room] + order
    want = max(depth(n) for n in rooms) * (0.5 + 0.5 * D)   # harder => gem farther
    gem_room = min(rooms, key=lambda n: (abs(depth(n) - want), n))
    chain = []
    n = gem_room
    while n is not None:
        chain.append(n)
        n = parent[n]
    chain.reverse()
    anchors = []
    for i, (rx, ry) in enumerate(chain):
        c = rx * ROOM_C + rng.rint(6, ROOM_C - 7)
        lo, hi = max(ry * ROOM_R + 5, 6), min(ry * ROOM_R + ROOM_R - 5, FLOOR - 5)
        anchors.append((c, rng.rint(lo, hi) if i else rng.rint(max(lo, hi - 6), hi)))
    if len(chain) == 1:
        anchors.append((rng.rint(6, C - 7), rng.rint(6, max(7, FLOOR - 12))))

    # ---- stage 2: critical path of reversible ledges ----------------------
    rng = substream(seed, attempt, "path")
    wmin = round(12 - 9 * D)                               # min take-off window required
    moves = [(g, d) for (g, d), w in window.items()
             if abs(d) <= 4 and w >= wmin and window.get((g, -d), 0) >= wmin]
    door_c = min(max(start_room[0] * ROOM_C + rng.rint(2, ROOM_C - 8), 1), C - 8)
    spawn_c = door_c + 3
    floor_keep = set(range(door_c - 2, spawn_c + 6))
    path = [Ledge(FLOOR, max(door_c - 2, 0), min(spawn_c + 8, C - 1))]

    def free(r0, r1, c0, c1, ignore=()):
        for r in range(max(r0, 0), min(r1, R - 1) + 1):
            row = grid[r]
            for c in range(max(c0, 0), min(c1, C - 1) + 1):
                if row[c] and not any(l.r == r and l.c0 <= c <= l.c1 for l in ignore):
                    return False
        return True

    def unreserved(r, c0, c1):
        return not any(reserved[r][c] for c in range(c0, c1 + 1))

    def reserve(r0, r1, c0, c1, journal=None):
        for r in range(max(r0, 0), min(r1, R - 1) + 1):
            row = reserved[r]
            for c in range(max(c0, 0), min(c1, C - 1) + 1):
                if not row[c]:
                    row[c] = 1
                    if journal is not None:
                        journal.append((r, c))

    def try_place(a: Ledge, sign, g, d, w):
        """Place a ledge reachable from ``a``; return (ledge, journal) or None."""
        rb = a.r - d
        if sign > 0:
            b0 = a.c1 + 1 + g
            b1 = b0 + w - 1
        else:
            b1 = a.c0 - 1 - g
            b0 = b1 - w + 1
        if b0 < 1 or b1 > C - 2 or rb < 5 or rb > FLOOR - 3:
            return None
        b = Ledge(rb, b0, b1)
        lo = a if a.r >= rb else b                             # lower ledge (larger row)
        need_hi = 3 if (a.r != rb and table.walkoff.get((g, -abs(d)), False)) else 7
        if sign > 0:
            za, zb = (a.c1 - 3, b0 + 1), (a.c1 - 1, min(b0 + 3, b1))
        else:
            za, zb = (b1 - 1, a.c0 + 3), (max(b1 - 3, b0), a.c0 + 1)
        zones = [(a.r, za[0], za[1], 7 if a is lo else need_hi),
                 (rb, zb[0], zb[1], 7 if b is lo else need_hi)]
        if not unreserved(rb, b0, b1):
            return None
        if not free(rb - 3, rb + 1, b0 - 1, b1 + 1, ignore=(a,)):
            return None                                        # air around the new ledge
        for zr, z0, z1, h_ in zones:
            if not free(zr - h_, zr - 1, z0, z1, ignore=(a, b)):
                return None                                    # jump corridor
        journal = []
        for c in range(b0, b1 + 1):
            grid[rb][c] = SOLID
        for zr, z0, z1, h_ in zones:
            reserve(zr - h_, zr - 1, z0, z1, journal)
        reserve(rb - 3, rb - 1, b0, b1, journal)
        return b, journal

    def undo(b: Ledge, journal):
        for c in range(b.c0, b.c1 + 1):
            grid[b.r][c] = EMPTY
        for r, c in journal:
            reserved[r][c] = 0

    def candidates(cur: Ledge, anchor):
        ac, ar = anchor
        dx, dr = ac - cur.mid, ar - cur.r
        pref = [2, 3, 4] if dr < -2 else [-2, -3, -4] if dr > 2 else [-1, 0, 1]
        toward = 1 if dx > 0 else -1
        out, seen_desc = [], set()
        for _ in range(48):
            sign = toward if (abs(dx) > 4 and rng.chance(0.75)) else rng.choice([-1, 1])
            pool = [m for m in moves if m[1] in pref] if rng.chance(0.75) else moves
            pool = pool or moves
            wts = [window[m] if D <= 0.3 else max(1, int((60 - window[m]) ** (1 + 2 * D)))
                   for m in pool]
            g, d = rng.weighted(pool, wts)
            w = rng.rint(3 if D > 0.6 else 4, 8 if D < 0.7 else 6)
            desc = (sign, g, d, w)
            if desc in seen_desc:
                continue
            seen_desc.add(desc)
            res = try_place(cur, *desc)
            if not res:
                continue
            b, journal = res
            undo(b, journal)
            score = abs(ac - b.mid) + 2 * abs(ar - b.r) + rng.below(6)
            out.append((score, len(out), desc))
            if len(out) == 6:
                break
        out.sort()
        return [desc for _, _, desc in out]

    budget, pops = 100 + 60 * len(anchors), 0
    frames = [[path[0], None, None, 0]]
    while True:
        cur, _, cands, ai = frames[-1]
        if len(frames) > 1 and max(abs(cur.mid - anchors[ai][0]), abs(cur.r - anchors[ai][1])) <= 5:
            ai += 1
            frames[-1][3] = ai
            frames[-1][2] = None
            cands = None
            if ai == len(anchors):
                break
        if cands is None:
            cands = frames[-1][2] = candidates(cur, anchors[ai])
        if not cands:
            if len(frames) == 1:
                raise GenFail("camino bloqueado")
            pops += 1                                          # backjumping
            for _ in range(min(1 + pops // 6, len(frames) - 1)):
                b_, journal, _, _ = frames.pop()
                undo(b_, journal)
            continue
        budget -= 1
        if budget < 0:
            raise GenFail("presupuesto de colocaciones agotado")
        b, journal = try_place(cur, *cands.pop(0))
        frames.append([b, journal, None, ai])
    path = [f[0] for f in frames]
    for f in frames[1:2]:
        lo_c, hi_c = min(path[0].c0, f[0].c0), max(path[0].c1, f[0].c1)
        floor_keep.update(range(lo_c - 1, hi_c + 2))
    goal = path[-1]
    gem_tile = (goal.mid, goal.r - 2)

    # ---- stage 3: hazards (never break reversibility) ---------------------
    rng = substream(seed, attempt, "hazards")
    if cfg.ledge_spikes and D > 0.2:
        for l in path[1:-1]:
            if l.c1 - l.c0 + 1 >= 7 and rng.chance(0.25 + 0.5 * D) and free(l.r - 8, l.r - 1, l.c0, l.c1):
                n = 1 if D < 0.6 else rng.rint(1, 2)
                c = l.mid - n // 2
                for k in range(n):
                    grid[l.r - 1][c + k] = SPIKE
                reserve(l.r - 8, l.r - 1, l.c0, l.c1)
    if cfg.floor_pits and D > 0.1:
        n_feat = int((C / ROOM_C) * (1 + 3 * D))
        c = 3
        while c < C - 8 and n_feat > 0:
            c += rng.rint(8, 16)
            wdt = rng.rint(2, 2 + int(2 * D))
            span = range(c - 5, c + wdt + 5)
            if c + wdt >= C - 3 or any(k in floor_keep for k in span):
                continue
            if not free(FLOOR - 8, FLOOR - 1, c - 4, c + wdt + 3):
                continue
            if rng.chance(0.5):
                for k in range(c, c + wdt):
                    grid[FLOOR][k] = grid[FLOOR + 1][k] = EMPTY
            else:
                for k in range(c, c + min(wdt, 3)):
                    grid[FLOOR - 1][k] = SPIKE
            reserve(FLOOR - 8, FLOOR - 1, c - 4, c + wdt + 3)
            n_feat -= 1

    # ---- stage 4: distractor ledges (dead ends, reachable and harmless) ----
    rng = substream(seed, attempt, "decor")
    n_dis = 0
    if cfg.distractors:
        for _ in range(int(len(path) * (0.3 + 0.7 * D))):
            a = rng.choice(path[1:]) if len(path) > 1 else path[0]
            g, d = rng.choice(moves)
            if try_place(a, rng.choice([-1, 1]), g, d, rng.rint(3, 6)):
                n_dis += 1

    # ---- stage 5: emit tiles + metadata -----------------------------------
    rows = ["".join(".#^"[v] for v in row) for row in grid]

    def put(r, c, ch):
        rows[r] = rows[r][:c] + ch + rows[r][c + 1:]

    if cfg.win_condition == 'exit_with_gem':
        put(FLOOR - 1, door_c, "D")
    put(FLOOR - 1, spawn_c, "P")
    put(gem_tile[1], gem_tile[0], "G")

    windows = [min(window[(max(b.c0 - a.c1, a.c0 - b.c1) - 1, a.r - b.r)],
                   window[(max(b.c0 - a.c1, a.c0 - b.c1) - 1, b.r - a.r)])
               for a, b in zip(path, path[1:])]
    pts = [(path[0].mid, path[0].r)] + [(l.mid, l.r) for l in path[1:]]
    manhattan = sum(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in zip(pts, pts[1:]))
    trips = 2 if cfg.win_condition == 'exit_with_gem' else 1
    est = manhattan * TILE_SIZE * trips / 1.5              # move_speed
    max_steps = max(1800, int(3 * est) // 100 * 100)

    data = {
        'version': 3,
        'name': f'gen-v{GENERATOR_VERSION}-{seed}',
        'tile_size': TILE_SIZE,
        'win_condition': cfg.win_condition,
        'max_steps': max_steps,
        'tiles': rows,
        'metadata': {
            'generator': {'version': GENERATOR_VERSION, 'seed': seed, 'attempt': attempt,
                          'config': _config_dict(cfg)},
            'waypoints': [[l.r, l.c0, l.c1] for l in path[1:]],
            'rooms': {'grid': [RC, RR], 'path': [list(n) for n in chain],
                      'tree': [[list(n), list(p)] for n, p in parent.items() if p is not None]},
            'stats': {'path_ledges': len(path) - 1, 'distractors': n_dis,
                      'min_window': min(windows), 'mean_window': round(sum(windows) / len(windows), 1),
                      'rooms_on_path': len(chain)},
        },
    }
    return Level.from_dict(data)


def level_to_dict(level: Level) -> dict:
    """Serialize a generated Level back to its canonical JSON dict (stable key order)."""
    return {
        'version': 3,
        'name': level.name,
        'tile_size': TILE_SIZE,
        'win_condition': level.win_condition,
        'max_steps': level.max_steps,
        'tiles': list(level.tiles),
        'metadata': level.metadata,
    }
