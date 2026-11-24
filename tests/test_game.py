import pytest
from dataclasses import replace

from game import Action, Game, Level


def make_game(max_steps=1800, **updates):
    data = dict(width=240, height=160, player_spawn=[16, 128],
                gem=[216, 132], platforms=[[0, 144, 240, 16]])
    data.update(updates)
    return Game(Level.from_dict(data), max_steps=max_steps)


@pytest.mark.parametrize('kind', ['dead', 'won', 'timeout'])
def test_terminal_priority_noop_and_exact_reset(kind):
    updates = {}
    if kind in ('won', 'dead'):
        updates['gem'] = [28, 132]
    if kind == 'dead':
        updates['hazards'] = [[28, 132, 8, 8]]
    game = make_game(max_steps=1, **updates)
    initial = game.get_state()
    result = game.step(Action.RIGHT)
    assert result.done and result.state.step_count == 1
    assert (result.dead, result.won, result.timeout) == tuple(
        kind == option for option in ('dead', 'won', 'timeout'))
    assert result.dead == (not result.state.alive)
    assert result.won == result.state.gem_collected
    assert result.state == game.get_state()
    assert initial.step_count == 0 and initial.player_x == 16
    for action in (Action.LEFT_JUMP, 999, None):
        assert game.step(action) == result
        assert game.get_state() == result.state and game.step_count == 1
    assert game.reset() == initial
    assert not any((game.is_done, game.is_dead, game.is_won, game.is_timeout))
    assert game.step_count == 0


def test_timeout_exact_threshold():
    game = make_game(max_steps=3)
    for step in range(1, 4):
        result = game.step(Action.NOOP)
        assert result.done == (step == 3)
        assert result.timeout == (step == 3)
        assert game.step_count == step


def test_no_timeout_when_disabled():
    game = make_game(max_steps=None)
    for _ in range(2000):
        assert not game.step(Action.NOOP).done
    assert game.step_count == 2000


def test_falling_below_world_causes_death_only_after_top_passes_height():
    game = make_game(platforms=[])
    for _ in range(100):
        result = game.step(Action.NOOP)
        assert result.dead == (result.state.player_y > game.level.height)
        if result.dead:
            assert result.done and not result.won and not result.timeout
            return
    pytest.fail('did not die below world')


def test_reset_mid_episode():
    game = make_game()
    initial = game.get_state()
    for _ in range(10):
        game.step(Action.RIGHT_JUMP)
    assert game.get_state() != initial
    assert game.reset() == initial


def test_exit_is_non_solid_and_does_not_win():
    game = make_game(exit=[28, 128])
    state = game.step(Action.RIGHT).state
    assert state.player_x == 17.5 and not state.won


def test_trigger_touching_edge_is_not_overlap():
    game = make_game(gem=[29.5, 132], hazards=[[29.5, 128, 8, 8]])
    assert not game.step(Action.RIGHT).done
    assert game.step(Action.RIGHT).dead


# ---- grounded is probed at reset -----------------------------------------

def test_reset_on_ground_probe_and_first_step_jump():
    game = make_game()
    state = game.get_state()
    assert state.grounded and state.velocity_y == 0
    # A jump issued on the very first step must take off: after the jump
    # impulse (-4.5) plus one gravity step (+0.25) vy == -4.25.
    result = game.step(Action.JUMP)
    assert result.state.velocity_y == pytest.approx(-4.25) and not result.state.grounded


def test_reset_in_the_air_probe_false():
    # Spawn elevated with nothing under the feet within 1 px.
    game = make_game(player_spawn=[16, 30], platforms=[[0, 144, 240, 16]])
    assert not game.get_state().grounded


def test_reset_requires_horizontal_overlap():
    # Player feet at y=144; a platform whose top face is exactly at foot
    # height but that starts at the player's right edge does not overlap.
    data = dict(width=240, height=160, player_spawn=[16, 128],
                gem=[216, 132], platforms=[[28, 144, 200, 16]])
    game = Game(Level.from_dict(data), max_steps=1800)
    assert not game.get_state().grounded
    # Same top face, but the platform really spans under the player.
    data['platforms'] = [[0, 144, 240, 16]]
    game = Game(Level.from_dict(data), max_steps=1800)
    assert game.get_state().grounded


def test_reset_fixed_point_noop_state_identical():
    game = make_game()
    initial = game.get_state()
    after = game.step(Action.NOOP).state
    # Rest is a fixed point from the first step: identical except step_count.
    assert after == replace(initial, step_count=1)


@pytest.mark.parametrize('number', range(1, 10))
def test_fixed_point_on_every_builtin_level(number):
    game = Game.from_level(f'levels/level_{number:03}.json', max_steps=None)
    initial = game.get_state()
    assert game.step(Action.NOOP).state == replace(initial, step_count=1)


def test_reset_level_applies_probe_to_new_level():
    game = make_game()
    assert game.get_state().grounded
    floating = Level.from_dict(dict(width=240, height=160, player_spawn=[16, 30],
                                    gem=[216, 10], platforms=[[0, 144, 240, 16]]))
    game.reset(level=floating)
    assert not game.get_state().grounded
    game.reset(level=make_game().level)
    assert game.get_state().grounded


def test_load_state_restores_saved_grounded_without_probing():
    level = Level.from_dict(dict(width=240, height=160, player_spawn=[16, 128],
                                 gem=[216, 132], platforms=[[0, 144, 240, 16]]))
    ground = Game(level, max_steps=1800)
    snap_grounded = ground.save_state()
    # A snapshot taken in the air (grounded False) restored into a game whose
    # probe would say grounded True: load_state must not re-probe.
    jumper = Game(level, max_steps=1800)
    jumper.step(Action.JUMP)
    assert not jumper.get_state().grounded
    snap_air = jumper.save_state()
    ground.load_state(snap_air)
    assert not ground.player.grounded and not ground.get_state().grounded
    ground.load_state(snap_grounded)
    assert ground.player.grounded and ground.get_state().grounded
