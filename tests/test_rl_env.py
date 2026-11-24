"""Gymnasium wrapper (rl/GemEnv): API conformity, rewards, frame skip and the
egocentric tile grid, checked against the engine it wraps."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pytest

gym = pytest.importorskip("gymnasium")
np = pytest.importorskip("numpy")

from gymnasium.utils.env_checker import check_env  # noqa: E402

from game import Action, Game, Level  # noqa: E402
from game.collision import overlaps  # noqa: E402
from game.constants import TILE_SIZE  # noqa: E402
from game.entities import Rect  # noqa: E402
from pcg import PRESETS, LevelDistribution  # noqa: E402
from rl import GRID_CHANNELS, VEC_FIELDS, GemEnv  # noqa: E402
from rl.env import DOOR, GEM, HAZARD, OUTSIDE, SOLID  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def solution_actions(n):
    sol = json.loads((ROOT / 'levels' / 'solutions' / f'level_{n:03d}.json').read_text())
    return [a for a, rep in sol['actions'] for _ in range(rep)]


# -- API ---------------------------------------------------------------------

@pytest.mark.parametrize('kwargs', [
    dict(levels=1),
    dict(levels=[4, 5, 6], frame_skip=1),
    dict(levels=7, obs_mode='pixels'),
    dict(levels=1, obs_mode='pixels', view='full_small'),
])
def test_passes_gymnasium_env_checker(kwargs):
    env = GemEnv(**kwargs)
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_spaces_shapes():
    env = GemEnv(levels=1, grid_size=(11, 7))
    obs, _ = env.reset(seed=0)
    assert obs['grid'].shape == (len(GRID_CHANNELS), 7, 11) and obs['grid'].dtype == np.uint8
    assert obs['vec'].shape == (len(VEC_FIELDS),) and obs['vec'].dtype == np.float32
    assert env.action_space.n == len(Action)
    pix = GemEnv(levels=1, obs_mode='pixels')
    assert pix.reset(seed=0)[0]['pixels'].shape == (160, 240, 3)
    pix.close()


@pytest.mark.parametrize('kwargs', [
    dict(obs_mode='voxels'), dict(grid_size=(30, 21)), dict(grid_size=(31, 0)),
    dict(frame_skip=0), dict(view='side'), dict(render_mode='human'),
])
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        GemEnv(levels=1, **kwargs)


def test_invalid_action_and_step_before_reset():
    env = GemEnv(levels=1)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    for bad in (6, -1, 2.0):
        with pytest.raises(ValueError):
            env.step(bad)
    env.step(np.int64(2))       # numpy scalars from vector envs are fine


def test_grid_mode_does_not_import_pygame():
    code = ("import sys; from rl import GemEnv; e = GemEnv(levels=7); e.reset(seed=0); "
            "[e.step(5) for _ in range(20)]; "
            "assert 'pygame' not in sys.modules, 'pygame imported'")
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


def test_engine_and_generator_do_not_import_gymnasium():
    code = ("import sys, game, pcg; "
            "assert 'gymnasium' not in sys.modules, 'gymnasium imported'")
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)


# -- episode semantics -------------------------------------------------------

@pytest.mark.parametrize('n', range(1, 10))
def test_reference_solution_wins_through_the_env(n):
    env = GemEnv(levels=n, frame_skip=1, gem_reward=0.5, max_steps=None)
    env.reset(seed=0)
    total, gem_events = 0.0, 0
    for i, a in enumerate(solution_actions(n), 1):
        _, reward, terminated, truncated, info = env.step(a)
        total += reward
        gem_events += info['gem_collected_event']
        if terminated or truncated:
            break
    assert terminated and not truncated and info['is_success']
    assert gem_events == 1
    two_stage = env.level.win_condition == 'exit_with_gem'
    assert total == (1.5 if two_stage else 1.0)


def test_frame_skip_matches_engine_and_stops_at_terminal():
    actions = solution_actions(4)                  # 220 steps, not a multiple of 8
    env = GemEnv(levels=4, frame_skip=8, max_steps=None)
    game = Game.from_level(ROOT / 'levels' / 'level_004.json', max_steps=None)
    env.reset(seed=0)
    # Hold each block's first action, mirroring it on a bare engine.
    i, steps = 0, []
    while True:
        a = actions[i]
        _, reward, terminated, truncated, info = env.step(a)
        for _ in range(info['engine_steps']):
            game.step(a)
        steps.append(info['engine_steps'])
        assert env.game.get_state() == game.get_state()
        i += 8
        if terminated or truncated or i >= len(actions):
            break
    assert all(s == 8 for s in steps[:-1])


def test_death_and_penalties():
    # Level 7, RIGHT only: dead at engine step 101 (tests/test_conformity.py).
    env = GemEnv(levels=7, frame_skip=1, death_penalty=1.0, step_penalty=0.001)
    env.reset(seed=0)
    total = 0.0
    for _ in range(200):
        _, reward, terminated, truncated, info = env.step(Action.RIGHT)
        total += reward
        if terminated:
            break
    assert terminated and not truncated and not info['is_success']
    assert info['step_count'] == 101
    assert total == pytest.approx(-1.0 - 101 * 0.001)


def test_timeout_is_truncation():
    # Level 4, RIGHT only: walks into the wall carrying the gem until timeout.
    env = GemEnv(levels=4, max_steps=100, frame_skip=7)
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert truncated and not terminated and reward == 0.0
    assert info['step_count'] == 100 and info['has_gem']


def test_gem_reward_not_paid_on_gem_only_levels():
    env = GemEnv(levels=1, frame_skip=1, gem_reward=0.5)
    env.reset(seed=0)
    total = sum(env.step(a)[1] for a in solution_actions(1))
    assert total == 1.0


# -- level sampling and determinism -----------------------------------------

def test_level_distribution_sampling_is_seeded():
    dist = LevelDistribution(PRESETS['tiny'], start_seed=0, num_levels=50)

    def names(seed):
        env = GemEnv(levels=dist)
        env.reset(seed=seed)
        out = [env.level]
        for _ in range(9):
            env.reset()
            out.append(env.level)
        return out

    a, b = names(123), names(123)
    assert all(x is y for x, y in zip(a, b))        # same cached Level objects
    assert len({id(x) for x in a}) > 1


def test_options_level_overrides_sampling():
    env = GemEnv(levels=1)
    env.reset(seed=0, options={'level': 9})
    assert env.level.height == 480
    env.reset(seed=0)
    assert env.level.height == 160


def test_same_seed_same_trajectory():
    def run():
        env = GemEnv(levels=LevelDistribution(PRESETS['small'], 0, 8))
        obs, _ = env.reset(seed=7)
        rng = np.random.default_rng(1)
        seq = [obs['vec'].tobytes() + obs['grid'].tobytes()]
        for _ in range(300):
            obs, _, term, trunc, _ = env.step(int(rng.integers(6)))
            seq.append(obs['vec'].tobytes() + obs['grid'].tobytes())
            if term or trunc:
                obs, _ = env.reset()
        return seq
    assert run() == run()


def test_sync_vector_env_autoresets():
    venv = gym.vector.SyncVectorEnv([lambda: GemEnv(levels=[1, 2, 3]) for _ in range(4)])
    obs, _ = venv.reset(seed=0)
    assert obs['grid'].shape[0] == 4
    rng = np.random.default_rng(0)
    for _ in range(600):
        obs, rew, term, trunc, _ = venv.step(rng.integers(6, size=4))
    assert rew.shape == (4,) and obs['vec'].shape == (4, len(VEC_FIELDS))
    venv.close()


# -- observation content -----------------------------------------------------

def brute_force_grid(env):
    """Independent per-cell rasterisation of the crop, straight from the engine."""
    cols, rows = env.grid_size
    level, game = env.level, env.game
    p = game.player
    pc = int((p.x + p.w / 2) // TILE_SIZE)
    pr = int((p.y + p.h / 2) // TILE_SIZE)
    wcols, wrows = level.width // TILE_SIZE, level.height // TILE_SIZE
    grid = np.zeros((len(GRID_CHANNELS), rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            wr, wc = pr - rows // 2 + i, pc - cols // 2 + j
            cell = Rect(wc * TILE_SIZE, wr * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            inside = 0 <= wr < wrows and 0 <= wc < wcols
            grid[OUTSIDE, i, j] = not inside
            grid[SOLID, i, j] = (not 0 <= wc < wcols     # side walls, any height
                                 or any(overlaps(cell, q) for q in level.platforms))
            grid[HAZARD, i, j] = any(overlaps(cell, h) for h in level.hazards)
            grid[GEM, i, j] = not game.has_gem and overlaps(cell, level.gem)
            grid[DOOR, i, j] = level.exit is not None and overlaps(cell, level.exit)
    return grid


@pytest.mark.parametrize('levels', [3, 7, 9, LevelDistribution(PRESETS['wide'], 0, 3)])
def test_grid_matches_brute_force_rasterisation(levels):
    env = GemEnv(levels=levels, frame_skip=3)
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    checked = 0
    for t in range(400):
        if t % 13 == 0:
            np.testing.assert_array_equal(obs['grid'], brute_force_grid(env))
            checked += 1
        obs, _, term, trunc, _ = env.step(int(rng.integers(6)))
        if term or trunc:
            np.testing.assert_array_equal(obs['grid'], brute_force_grid(env))
            obs, _ = env.reset()
    assert checked > 20


def test_grid_handles_player_above_and_below_the_world():
    level = Level.from_dict({
        'width': 240, 'height': 160, 'player_spawn': [100, 0], 'gem': [200, 140],
        'platforms': [[0, 150, 80, 10]]})
    env = GemEnv(levels=level, frame_skip=1)
    env.reset(seed=0)
    env.game.player.y = -60.0               # far above the open top
    np.testing.assert_array_equal(env._grid(), brute_force_grid(env))
    env.game.player.y = 175.0               # fallen past the bottom
    np.testing.assert_array_equal(env._grid(), brute_force_grid(env))


def test_vector_features_and_goal_vector_switch():
    env = GemEnv(levels=4, frame_skip=1)
    obs, _ = env.reset(seed=0)
    v = dict(zip(VEC_FIELDS, obs['vec']))
    lv, p = env.level, env.game.player
    assert v['grounded'] == 1 and v['has_gem'] == 0 and v['needs_door'] == 1
    assert v['gem_dx'] == pytest.approx((lv.gem.x + 4 - p.x - 6) / 240)
    assert v['door_dx'] == pytest.approx((lv.exit.x + 6 - p.x - 6) / 240)
    assert v['time_left'] == 1.0
    obs, *_ = env.step(Action.RIGHT)
    assert dict(zip(VEC_FIELDS, obs['vec']))['vx'] == 1.0
    blind = GemEnv(levels=4, goal_vector=False)
    vb = dict(zip(VEC_FIELDS, blind.reset(seed=0)[0]['vec']))
    assert all(vb[k] == 0 for k in ('gem_dx', 'gem_dy', 'door_dx', 'door_dy'))


def test_render_rgb_array_shows_whole_world():
    env = GemEnv(levels=8, render_mode='rgb_array')
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (160, 720, 3) and frame.dtype == np.uint8
    env.close()


@pytest.mark.parametrize('n', [3, 4, 7])
def test_distance_shaping_telescopes_over_an_episode(n):
    # Potential-based shaping: the summed shaping term over an episode equals
    # c * (phi(final) - phi(initial)), whatever the trajectory.
    c = 0.3
    shaped = GemEnv(levels=n, frame_skip=1, distance_shaping=c, max_steps=None)
    plain = GemEnv(levels=n, frame_skip=1, max_steps=None)
    shaped.reset(seed=0)
    plain.reset(seed=0)
    phi0 = shaped.potential()
    diff = 0.0
    for a in solution_actions(n):
        _, rs, term, _, _ = shaped.step(a)
        _, rp, *_ = plain.step(a)
        diff += rs - rp
        if term:
            break
    assert term
    assert diff == pytest.approx(c * (shaped.potential() - phi0))


def test_potential_at_gem_pickup_only_rises_a_little():
    env = GemEnv(levels=4, frame_skip=1)
    env.reset(seed=0)
    phis = []
    for a in solution_actions(4):
        phis.append((env.game.has_gem, env.potential()))
        env.step(a)
        if env.game.has_gem and not phis[-1][0]:
            phis.append((True, env.potential()))
            break
    # Picking up the gem never lowers the potential (triangle inequality), and
    # raises it by at most twice the centre offset at first contact:
    # 2 * hypot((6 + 4) / 240, (8 + 4) / 160) ~= 0.17 screens.
    jump = phis[-1][1] - phis[-2][1]
    assert 0 <= jump <= 2 * math.hypot(10 / 240, 12 / 160)
