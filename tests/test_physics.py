import pytest

from game import Action, Game, Level


def make_game(**updates):
    data = dict(width=240, height=160, player_spawn=[16, 30], gem=[216, 10],
                platforms=[])
    data.update(updates)
    return Game(Level.from_dict(data), max_steps=None)


@pytest.mark.parametrize('action,dx,jump', [(0, 0, False), (1, -1.5, False),
    (2, 1.5, False), (3, 0, True), (4, -1.5, True), (5, 1.5, True)])
def test_air_control_gravity_and_no_double_jump(action, dx, jump):
    game = make_game()
    state = game.step(action).state
    assert state.player_x == 16 + dx
    assert state.velocity_x == dx
    assert state.velocity_y == 0.25 and state.player_y == 30.25
    assert not state.grounded
    next_state = game.step(Action.NOOP).state
    assert next_state.velocity_x == 0
    assert next_state.velocity_y == 0.5
    assert state.step_count == 1 and next_state.step_count == 2


def test_gravity_saturates_at_max_fall_speed():
    game = make_game(height=1000)
    for _ in range(50):
        state = game.step(Action.NOOP).state
        assert state.velocity_y <= 4
    assert state.velocity_y == 4


@pytest.mark.parametrize('field,value', [
    ('gravity', 0), ('gravity', float('nan')), ('gravity', float('inf')),
    ('move_speed', -1), ('move_speed', 12), ('move_speed', True),
    ('jump_velocity', 1), ('jump_velocity', -12), ('max_fall_speed', 12),
    ('max_fall_speed', '4'), ('gravity', 12),
])
def test_physics_config_validation(field, value):
    from game.physics import PhysicsConfig
    with pytest.raises(ValueError, match=field):
        PhysicsConfig(**{field: value})


def test_custom_physics():
    from game.physics import PhysicsConfig
    game = Game(make_game().level, physics=PhysicsConfig(gravity=0.5, move_speed=2))
    state = game.step(Action.RIGHT).state
    assert state.player_x == 18 and state.player_y == 30.5


def test_first_step_jump_from_probed_reset():
    # After reset on the floor, JUMP on the very
    # first step applies the impulse: vy = -4.5 + 0.25 == -4.25, and the
    # player rises to y = 128 - 4.25.
    game = Game(Level.from_dict(dict(width=240, height=160, player_spawn=[16, 128],
                                     gem=[216, 132], platforms=[[0, 144, 240, 16]])),
                max_steps=None)
    state = game.step(Action.JUMP).state
    assert state.velocity_y == pytest.approx(-4.25)
    assert state.player_y == pytest.approx(128 - 4.25)
    assert not state.grounded


@pytest.mark.parametrize('action', [-1, 6, None, '2', 2.0, True, []])
def test_invalid_action_does_not_advance(action):
    game = make_game()
    before = game.get_state()
    with pytest.raises(ValueError):
        game.step(action)
    assert game.get_state() == before
