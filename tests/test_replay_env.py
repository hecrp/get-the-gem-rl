"""GemEnv <-> game.replay.simulate equivalence and info['level_ref'] (spec 13 tests 7-8)."""
import json
from pathlib import Path

import numpy as np
import pytest

from game import Level
from game.level import level_hash
from game.replay import rle_encode, simulate
from pcg import PRESETS, GENERATOR_VERSION, LevelDistribution, LevelPack, generate_level
from rl.env import BUILTIN_LEVELS, GemEnv

ROOT = Path(__file__).resolve().parents[1]


def _episode(env, rng, options=None, seed=None):
    obs, info = env.reset(seed=seed, options=options)
    actions = []
    done = False
    while not done:
        a = int(rng.integers(6))
        actions.append(a)
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    return actions, info


def _check(env, actions, info):
    game = env.game
    traj = simulate(game.level, rle_encode(actions), frame_skip=env.frame_skip,
                    max_steps=game.max_steps, record=False)
    assert traj.env_steps == len(actions)
    assert traj.engine_steps == game.step_count == info['step_count']
    assert (traj.result == 'won') == game.is_won
    assert (traj.result == 'dead') == game.is_dead
    assert (traj.result == 'timeout') == game.is_timeout
    assert traj.end_reason == game.end_reason
    assert traj.final_state == game.get_state()


@pytest.mark.parametrize('frame_skip', [1, 4])
def test_env_replay_equivalence(frame_skip):
    rng = np.random.default_rng(2026 + frame_skip)
    n = 0
    # Built-in levels (random policy; biased to RIGHT moves so episodes vary).
    env = GemEnv(levels=list(range(1, 10)), frame_skip=frame_skip, max_steps=400)
    for ep in range(60):
        actions, info = _episode(env, rng, seed=ep)
        _check(env, actions, info)
        n += 1
    # Generated levels.
    dist = LevelDistribution(PRESETS['tiny'], 500, 20)
    env = GemEnv(levels=dist, frame_skip=frame_skip, max_steps=300)
    for ep in range(40):
        actions, info = _episode(env, rng, seed=ep)
        _check(env, actions, info)
        n += 1
    assert n == 100            # x 2 frame_skip values = 200 episodes


def test_level_ref_builtin_and_list():
    env = GemEnv(levels=7)
    _, info = env.reset(seed=0)
    assert info['level_ref'] == {'kind': 'builtin', 'n': 7}
    _, _, _, _, info = env.step(0)
    assert info['level_ref'] == {'kind': 'builtin', 'n': 7}
    env = GemEnv(levels=[2, 5, 8])
    seen = set()
    for s in range(20):
        _, info = env.reset(seed=s)
        ref = info['level_ref']
        assert ref['kind'] == 'builtin'
        assert env.level == Level.from_json(BUILTIN_LEVELS[ref['n'] - 1])
        seen.add(ref['n'])
    assert seen == {2, 5, 8}
    env = GemEnv(levels=[str(BUILTIN_LEVELS[2])])
    assert env.reset()[1]['level_ref'] == {'kind': 'builtin', 'n': 3}


def test_level_ref_distribution_and_regeneration():
    dist = LevelDistribution(PRESETS['tiny'], 1000, 50)
    env = GemEnv(levels=dist)
    for s in range(5):
        _, info = env.reset(seed=s)
        ref = info['level_ref']
        assert ref['kind'] == 'pcg' and ref['preset'] == 'tiny'
        assert ref['generator_version'] == GENERATOR_VERSION
        assert 1000 <= ref['seed'] < 1050
        from pcg import GenConfig
        assert generate_level(ref['seed'], GenConfig(**ref['config'])) == env.level
        json.dumps(ref)
    unbounded = GemEnv(levels=LevelDistribution(PRESETS['tiny'], 0, 0))
    ref = unbounded.reset(seed=3)[1]['level_ref']
    assert ref['kind'] == 'pcg' and generate_level(ref['seed'], PRESETS['tiny']) == unbounded.level


def test_level_ref_does_not_change_rng_or_levels():
    a = GemEnv(levels=LevelDistribution(PRESETS['tiny'], 0, 100))
    b = GemEnv(levels=LevelDistribution(PRESETS['tiny'], 0, 100))
    for s in range(5):
        a.reset(seed=s)
        b.reset(seed=s)
        assert a.level is not None and a.level == b.level
        assert a.np_random.bit_generator.state == b.np_random.bit_generator.state


def test_level_ref_pack(tmp_path):
    pack = tmp_path / 'p.jsonl'
    with pack.open('w') as fh:
        for seed in range(3):
            fh.write(json.dumps(generate_level(seed, PRESETS['tiny']).to_dict()) + '\n')
    env = GemEnv(levels=LevelPack(pack))
    _, info = env.reset(seed=0)
    ref = info['level_ref']
    assert ref['kind'] == 'custom' and ref['hash'] == level_hash(env.level)
    assert ref['pack'] == 'p.jsonl' and 0 <= ref['index'] < 3


def test_level_ref_loose_level_and_options():
    custom = Level.from_dict({'width': 240, 'height': 160, 'player_spawn': [16, 128],
                              'gem': [200, 132], 'platforms': [[0, 144, 240, 16]]})
    env = GemEnv(levels=custom)
    ref = env.reset()[1]['level_ref']
    assert ref == {'kind': 'custom', 'hash': level_hash(custom)}
    # A Level equal to a built-in is recognised as such.
    env = GemEnv(levels=Level.from_json(BUILTIN_LEVELS[0]))
    assert env.reset()[1]['level_ref'] == {'kind': 'builtin', 'n': 1}
    # A generated Level carries its generator provenance.
    gen = generate_level(77, PRESETS['small'])
    env = GemEnv(levels=gen)
    ref = env.reset()[1]['level_ref']
    assert ref['kind'] == 'pcg' and ref['seed'] == 77 and ref['preset'] == 'small'
    # options={'level': ...} overrides the source.
    env = GemEnv(levels=1)
    assert env.reset(options={'level': 4})[1]['level_ref'] == {'kind': 'builtin', 'n': 4}
    assert env.reset(options={'level': custom})[1]['level_ref']['kind'] == 'custom'
    assert env.reset()[1]['level_ref'] == {'kind': 'builtin', 'n': 1}
