"""game.replay: RLE, simulate(), events and the reference replays (spec 13 §92 E-2/E-3)."""
import random
from pathlib import Path

import pytest

from game import Game, Level
from game.replay import (DEAD, GEM_COLLECTED, WON, reference_replays, rle_decode,
                         rle_encode, simulate, simulate_replay)

ROOT = Path(__file__).resolve().parents[1]
LEVELS = ROOT / 'levels'


def _level(n):
    return Level.from_json(LEVELS / f'level_{n:03d}.json')


# -- RLE -----------------------------------------------------------------------

def test_rle_round_trip_random():
    rng = random.Random(1234)
    for _ in range(1000):
        seq = [rng.randrange(6) for _ in range(rng.randrange(0, 200))]
        # Long runs too.
        if rng.random() < 0.3:
            seq += [rng.randrange(6)] * rng.randrange(1, 500)
        pairs = rle_encode(seq)
        assert list(rle_decode(pairs)) == seq
        assert all(r >= 1 for _, r in pairs)
        assert all(a != b for (a, _), (b, _) in zip(pairs, pairs[1:]))


def test_rle_edge_cases():
    assert rle_encode([]) == []
    assert list(rle_decode([])) == []
    assert rle_encode([4]) == [[4, 1]]
    assert list(rle_decode([[5, 1]])) == [5]
    big = rle_decode([[2, 10 ** 9]])          # lazy: never materialised
    assert next(big) == 2


@pytest.mark.parametrize('pairs', [
    [[6, 1]], [[-1, 1]], [[1.0, 1]], [[True, 1]], [['1', 1]],
    [[1, 0]], [[1, -2]], [[1, 1.5]], [[1, True]], [[1]], [1], [[1, 2, 3]],
])
def test_rle_decode_rejects_invalid(pairs):
    with pytest.raises(ValueError):
        list(rle_decode(pairs))


@pytest.mark.parametrize('actions', [[6], [1.0], [True], ['x']])
def test_rle_encode_rejects_invalid(actions):
    with pytest.raises(ValueError):
        rle_encode(actions)


# -- simulate --------------------------------------------------------------------

def test_reference_solutions_win_with_simulate():
    replays = reference_replays(LEVELS / 'solutions')
    assert [r['n'] for r in replays] == list(range(1, 10))
    for r in replays:
        level = _level(r['n'])
        traj = simulate(level, r['actions'], frame_skip=1)
        assert traj.result == 'won' and traj.end_reason == 'won', r['id']
        assert r['outcome']['result'] == 'won'
        assert r['id'] == f"_reference/level_{r['n']:03d}"
        assert r['policy'] == 'reference' and r['env']['frame_skip'] == 1
        assert traj.engine_steps == r['outcome']['engine_steps']
        assert len(traj.xy) == 2 * (traj.engine_steps + 1)
        assert len(traj.flags) == traj.engine_steps + 1
        assert traj.flags[-1] & WON and traj.flags[-1] & GEM_COLLECTED
        assert traj.win_step == traj.engine_steps
        assert traj.gem_step is not None and traj.gem_step <= traj.win_step


def test_simulate_is_deterministic_and_matches_game():
    level = _level(9)
    rng = random.Random(7)
    rle = rle_encode(rng.randrange(6) for _ in range(500))
    a = simulate(level, rle, frame_skip=4)
    b = simulate(level, rle, frame_skip=4)
    assert a == b if not a.events else (a.xy, a.flags, a.events) == (b.xy, b.flags, b.events)
    game = Game(level)
    for act in rle_decode(rle):
        for _ in range(4):
            res = game.step(act)
            if res.done:
                break
        if game.is_done:
            break
    assert a.final_state == game.get_state()
    assert a.engine_steps == game.step_count


def test_simulate_record_false_same_summary():
    level = _level(7)
    rle = reference_replays(LEVELS / 'solutions')[6]['actions']
    full = simulate(level, rle)
    lite = simulate(level, rle, record=False)
    assert lite.xy == () and lite.flags == ()
    assert full.outcome() == lite.outcome() and full.events == lite.events


def test_simulate_incomplete_and_trailing_actions():
    level = _level(1)
    short = simulate(level, [[2, 10]])
    assert short.result == 'incomplete' and short.end_reason is None
    assert short.env_steps == 10 and short.engine_steps == 10
    # Actions after the terminal step are ignored.
    long = simulate(level, [[2, 126], [1, 50]])
    assert long.result == 'won' and long.env_steps == 126


def test_simulate_frame_skip_stops_at_terminal():
    level = _level(1)
    traj = simulate(level, [[2, 40]], frame_skip=4)   # wins at engine step 126
    assert traj.result == 'won' and traj.engine_steps == 126
    assert traj.env_steps == 32                      # ceil(126 / 4)


def test_simulate_events_death_by_hazard_and_fall():
    # Level 3 ("Pit"): walking right falls into the pit.
    traj = simulate(_level(3), [[2, 400]])
    assert traj.result == 'dead' and traj.end_reason == 'fall'
    dead = [e for e in traj.events if e['type'] == 'dead']
    assert dead and dead[0]['cause'] == 'fall' and 'hazard' not in dead[0]
    assert traj.death_xy is not None and traj.death_xy[1] > _level(3).height
    assert traj.flags[-1] & DEAD
    # Level 8 ("Long haul"): holding RIGHT_JUMP lands on its second spike strip.
    level = _level(8)
    t = simulate(level, [[5, 600]])
    assert t.end_reason == 'hazard' and t.result == 'dead'
    ev = [e for e in t.events if e['type'] == 'dead'][0]
    assert ev['cause'] == 'hazard' and ev['hazard'] == 1
    # Holding LEFT walks back into the first one.
    t = simulate(level, [[1, 600]])
    assert t.end_reason == 'hazard'
    assert [e for e in t.events if e['type'] == 'dead'][0]['hazard'] == 0


def test_simulate_timeout_event():
    traj = simulate(_level(1), [[0, 2000]], max_steps=50)
    assert traj.result == 'timeout' and traj.end_reason == 'timeout'
    assert traj.events[-1] == {'t': 50, 'type': 'timeout'}


def test_simulate_validates():
    with pytest.raises(ValueError):
        simulate(_level(1), [[9, 1]])
    with pytest.raises(ValueError):
        simulate(_level(1), [[1, 1]], frame_skip=0)


def test_simulate_replay_uses_env_block():
    r = reference_replays(LEVELS / 'solutions')[3]
    t = simulate_replay(r, _level(4))
    assert t.result == 'won'
    r2 = dict(r, env={'frame_skip': 1, 'max_steps': 100, 'physics': None})
    assert simulate_replay(r2, _level(4)).result == 'timeout'


# -- Game.end_reason (table) -------------------------------------------------------

def _run(level, actions, max_steps=None):
    game = Game(level, max_steps=max_steps)
    for a in actions:
        game.step(a)
        if game.is_done:
            break
    return game


def test_end_reason_table():
    assert Game(_level(1)).end_reason is None                      # running
    assert _run(_level(1), [2] * 200).end_reason == 'won'          # gem win
    assert _run(_level(3), [2] * 400).end_reason == 'fall'         # fell in the pit
    assert _run(_level(1), [0] * 100, max_steps=20).end_reason == 'timeout'
    g = _run(_level(8), [5] * 600)                                 # spikes
    assert g.is_dead and g.player.y <= g.level.height and g.end_reason == 'hazard'


def test_end_reason_precedence():
    # Dead wins over everything; won over timeout (same step).
    level = _level(1)
    g = Game(level, max_steps=126)              # gem reached exactly at step 126
    for _ in range(126):
        g.step(2)
    assert g.is_won and not g.is_timeout and g.end_reason == 'won'
    g = Game(level)
    g.player.alive = False
    g._won = g._timeout = True
    assert g.end_reason == 'hazard'
    g.player.y = level.height + 1
    assert g.end_reason == 'fall'
    g = Game(level)
    g._won = g._timeout = True
    assert g.end_reason == 'won'
