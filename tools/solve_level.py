#!/usr/bin/env python3
"""Greedy best-first solver that validates a level is winnable and emits a
reproducible reference solution.

Not part of the game: it is a content-validation tool. It drives the real
``Game`` engine via ``save_state``/``load_state`` with ``max_steps=None`` and
never imports anything from ``rendering/`` or ``play.py``.

Guided mode: if the level carries ``metadata.waypoints``, they
are used as sub-goals, which makes solving near-linear in the critical-path
length even for 1920x1280 worlds. ``k = 3`` steps per action by default so the
solver can hit take-off windows as narrow as 2-3 frames.

Usage:
    python -m tools.solve_level levels/level_007.json --out levels/solutions/level_007.json

Exit status: 0 if a solution was found, 1 otherwise.
"""
import argparse
import heapq
import json
import sys
from pathlib import Path

from game import Action, Game, Level
from game.constants import PLAYER_H, PLAYER_W, TILE_SIZE

WAYPOINT_BIG = 100_000   # dominates any Manhattan distance while a waypoint is pending


def _load(level):
    return level if isinstance(level, Level) else Level.from_json(level)


def solve(level, limit=3_000_000, k=3, guided=True, start_state=None):
    """Return ``(sequence, nodes)``, or ``(None, nodes)`` if no solution found.

    ``level`` is a ``Level`` or a path. ``guided`` enables waypoint sub-goals
    when the level carries ``metadata.waypoints``. ``start_state`` (a snapshot
    from the same level) starts the search mid-episode instead of at spawn.
    """
    level = _load(level)
    game = Game(level, max_steps=None)
    if start_state is not None:
        game.load_state(start_state)
    waypoints = level.metadata.get('waypoints', []) if guided else []
    n_wp = len(waypoints)

    def state_key(g):
        return (g.player.x, g.player.y, g.player.vy, g.player.grounded, g.has_gem)

    def on_waypoint(g, wp):
        r, c0, c1 = wp
        return (g.player.grounded and g.player.y + PLAYER_H == r * TILE_SIZE
                and g.player.x + PLAYER_W > c0 * TILE_SIZE
                and g.player.x < (c1 + 1) * TILE_SIZE)

    def heuristic(g, i):
        if g.has_gem:
            if level.exit is None or level.win_condition == 'gem':
                return 0
            return abs(g.player.x - level.exit.x) + abs(g.player.y - level.exit.y)
        if i < n_wp:
            r, c0, c1 = waypoints[i]
            tx = (c0 + c1 + 1) * TILE_SIZE / 2 - PLAYER_W / 2
            ty = r * TILE_SIZE - PLAYER_H
        else:
            tx, ty = level.gem.x, level.gem.y
        return (n_wp - i + 1) * WAYPOINT_BIG + abs(g.player.x - tx) + abs(g.player.y - ty)

    start = game.save_state()
    insert = 0
    heap = [(heuristic(game, 0), insert, start, 0, ())]
    visited = {(state_key(game), 0)}
    nodes = 0

    while heap:
        if nodes >= limit:
            return None, nodes
        _, _, snap, i, seq = heapq.heappop(heap)
        nodes += 1
        for ai in range(6):
            game.load_state(snap)
            mac = ()
            won = dead = False
            for _ in range(k):
                result = game.step(ai)
                mac += (ai,)
                if result.won:
                    won = True
                    break
                if result.dead:
                    dead = True
                    break
            if won:
                return seq + mac, nodes
            if dead:
                continue
            j = i
            while j < n_wp and not game.has_gem and on_waypoint(game, waypoints[j]):
                j += 1
            if game.has_gem:
                j = n_wp
            key = (state_key(game), j)
            if key in visited:
                continue
            visited.add(key)
            insert += 1
            heapq.heappush(heap, (heuristic(game, j), insert, game.save_state(), j, seq + mac))
    return None, nodes


def _rle(sequence):
    out = []
    for a in sequence:
        if out and out[-1][0] == a:
            out[-1][1] += 1
        else:
            out.append([a, 1])
    return out


def _metrics(level, sequence):
    game = Game(_load(level), max_steps=None, seed=0)
    gem_at = None
    for i, a in enumerate(sequence, 1):
        result = game.step(a)
        if gem_at is None and result.state.gem_collected:
            gem_at = i
    return game.step_count, gem_at


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('level', help='path to a level JSON file')
    parser.add_argument('--out', help='write the solution here (default: stdout)')
    parser.add_argument('--limit', type=int, default=3_000_000,
                        help='max expanded nodes (default 3,000,000)')
    parser.add_argument('--k', type=int, default=3,
                        help='hold each action for k steps (default 3)')
    parser.add_argument('--no-guided', action='store_true',
                        help='ignore metadata.waypoints even when present')
    args = parser.parse_args(argv)

    sequence, nodes = solve(args.level, limit=args.limit, k=args.k,
                            guided=not args.no_guided)
    if sequence is None:
        print(f'{Path(args.level).name}: no solution found after {nodes} nodes',
              file=sys.stderr)
        return 1

    total_steps, gem_at = _metrics(args.level, sequence)
    solution = {
        'level': Path(args.level).name,
        'format': 'rle [action, repeat]',
        'total_steps': total_steps,
        'gem_collected_at_step': gem_at,
        'actions': _rle(sequence),
    }
    text = json.dumps(solution)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
        print(f'{out}: solved in {total_steps} steps ({nodes} nodes)', file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
