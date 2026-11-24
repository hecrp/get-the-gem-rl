"""Keyboard/CLI tests and interactive loop smoke using SDL's dummy display."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from pathlib import Path

import pygame
import pytest

from game import Action, Game, Level


@pytest.mark.parametrize("left,right,jump,expected", [
    (False, False, False, Action.NOOP),
    (True, False, False, Action.LEFT),
    (False, True, False, Action.RIGHT),
    (False, False, True, Action.JUMP),
    (True, False, True, Action.LEFT_JUMP),
    (False, True, True, Action.RIGHT_JUMP),
    (True, True, False, Action.NOOP),
    (True, True, True, Action.JUMP),
])
def test_keys_to_action(left, right, jump, expected):
    from play import keys_to_action
    assert keys_to_action(left, right, jump) is expected


@pytest.mark.parametrize("argv,index,scale", [([], 0, None), (["--level", "2"], 1, None),
                                                 (["--level", "3", "--scale", "4"], 2, 4)])
def test_cli_builtin_paths_are_independent_of_cwd(monkeypatch, tmp_path, argv, index, scale):
    import play
    monkeypatch.chdir(tmp_path)
    args = play.parse_args(argv)
    assert args.level == Path(play.__file__).resolve().parent / "levels" / f"level_{index + 1:03d}.json"
    assert args.scale == scale


def test_cli_accepts_custom_path_relative_to_cwd(monkeypatch, tmp_path):
    import play
    monkeypatch.chdir(tmp_path)
    assert play.parse_args(["--level", "my_map.json"]).level == tmp_path / "my_map.json"


@pytest.mark.parametrize("scale", ["0", "-2", "1.5", "abc"])
def test_cli_rejects_invalid_scale(scale, capsys):
    import play
    with pytest.raises(SystemExit) as exc:
        play.parse_args(["--scale", scale])
    assert exc.value.code == 2
    assert "scale" in capsys.readouterr().err


@pytest.fixture
def flat_game():
    return Game(Level.from_dict({
        "width": 240, "height": 160, "player_spawn": [16, 128],
        "gem": [210, 128], "platforms": [[0, 144, 240, 16]], "hazards": [],
    }), max_steps=None)


@pytest.fixture
def event_queue():
    pygame.display.init()
    pygame.event.clear()
    yield
    pygame.display.quit()


def test_cli_view_flag_defaults_full_and_validates():
    import play
    assert play.parse_args([]).view == "full"
    assert play.parse_args(["--view", "follow"]).view == "follow"
    with pytest.raises(SystemExit):
        play.parse_args(["--view", "bogus"])


def test_auto_scale_fits_1280x800_with_minimum_1():
    import play
    assert play._auto_scale(240, 160) == 5       # 1280//240, 800//160
    assert play._auto_scale(480, 320) == 2       # 1280//480, 800//320
    assert play._auto_scale(2000, 2000) == 1     # floor at 1


def test_v_key_toggles_renderer_view(flat_game, event_queue):
    import play
    from rendering import PygameRenderer, ViewConfig, ViewMode
    renderer = PygameRenderer(headless=True,
                              view=ViewConfig(mode=ViewMode.FULL, canvas_size=None))
    assert renderer.view.mode is ViewMode.FULL
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
    game, path, running = play.process_events(flat_game, play.BUILTIN_LEVELS[0],
                                              pygame.event.get(), renderer)
    assert renderer.view.mode is ViewMode.FOLLOW
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_v))
    play.process_events(game, path, pygame.event.get(), renderer)
    assert renderer.view.mode is ViewMode.FULL


def test_reset_uses_keydown_from_actual_event_queue(flat_game, event_queue):
    import play
    initial = flat_game.get_state()
    flat_game.step(Action.RIGHT)
    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame.K_r))
    game, path, running = play.process_events(flat_game, play.BUILTIN_LEVELS[0], pygame.event.get())
    assert game.get_state() != initial
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    game, path, running = play.process_events(game, path, pygame.event.get())
    assert game is flat_game
    assert game.get_state() == initial
    assert running


@pytest.mark.parametrize("event", [pygame.event.Event(pygame.QUIT),
                                  pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)])
def test_quit_event_stops_without_reset(flat_game, event_queue, event):
    import play
    flat_game.step(Action.RIGHT)
    before = flat_game.get_state()
    pygame.event.post(event)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    game, path, running = play.process_events(flat_game, play.BUILTIN_LEVELS[0], pygame.event.get())
    assert not running
    assert game.get_state() == before


@pytest.mark.parametrize("key,index", [(pygame.K_1, 0), (pygame.K_2, 1), (pygame.K_3, 2)])
def test_number_keys_load_real_levels_without_timeout(flat_game, event_queue, key, index):
    import play
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    game, path, running = play.process_events(flat_game, play.BUILTIN_LEVELS[0], pygame.event.get())
    assert game is not flat_game
    assert path == play.BUILTIN_LEVELS[index]
    assert game.level == Level.from_json(path)
    assert game.max_steps is None
    assert game.step_count == 0
    assert running


@pytest.mark.parametrize("start,index", [(0, 1), (1, 2), (2, 3), (None, 0)])
def test_next_level_cycles_and_custom_path_starts_at_first(flat_game, event_queue, start, index):
    import play
    path = Path("/custom.json") if start is None else play.BUILTIN_LEVELS[start]
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
    game, path, running = play.process_events(flat_game, path, pygame.event.get())
    assert path == play.BUILTIN_LEVELS[index]
    assert game is not flat_game
    assert game.level == Level.from_json(path)
    assert game.max_steps is None
    assert running


@pytest.mark.parametrize("pressed,expected", [
    ({pygame.K_d: True}, Action.RIGHT), ({pygame.K_RIGHT: True}, Action.RIGHT),
    ({pygame.K_a: True}, Action.LEFT), ({pygame.K_LEFT: True}, Action.LEFT),
    ({pygame.K_SPACE: True}, Action.JUMP),
    ({pygame.K_a: True, pygame.K_SPACE: True}, Action.LEFT_JUMP),
    ({pygame.K_d: True, pygame.K_SPACE: True}, Action.RIGHT_JUMP),
    ({pygame.K_a: True, pygame.K_RIGHT: True}, Action.NOOP),
    ({pygame.K_LEFT: True, pygame.K_d: True, pygame.K_SPACE: True}, Action.JUMP),
])
def test_main_steps_once_per_frame_and_presents_real_dummy_window(monkeypatch, tmp_path, pressed, expected):
    import play
    from collections import defaultdict
    from rendering import PygameRenderer
    monkeypatch.chdir(tmp_path)
    pygame.display.quit()
    frames, actions, ticks = [], [], []
    original_step = Game.step
    original_present = PygameRenderer.present
    def step(game, action):
        actions.append(action)
        return original_step(game, action)
    def present(renderer, game):
        original_present(renderer, game)
        frames.append(game.get_state())
        assert game.max_steps is None
        assert pygame.display.get_surface().get_size() == (480, 320)
        if len(frames) == 3:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        assert len(frames) <= 3
    class Clock:
        def tick(self, fps):
            ticks.append(fps)
            return 999  # must not influence the number of logical steps
    monkeypatch.setattr(Game, "step", step)
    monkeypatch.setattr(PygameRenderer, "present", present)
    monkeypatch.setattr(pygame.time, "Clock", Clock)
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: defaultdict(bool, pressed))
    assert play.main(["--scale", "2"]) == 0
    assert [frame.step_count for frame in frames] == [1, 2, 3]
    assert actions == [expected] * 3
    assert ticks == [60] * 3
    assert not pygame.display.get_init()


@pytest.mark.parametrize("after_window", [False, True])
def test_main_finally_closes_on_exception(monkeypatch, after_window):
    import play
    from rendering import PygameRenderer
    original_present = PygameRenderer.present
    original_close = PygameRenderer.close
    closed = []
    def close(renderer):
        closed.append(renderer)
        original_close(renderer)
    def fail(*args):
        raise RuntimeError("injected failure")
    def present(renderer, game):
        original_present(renderer, game)
        fail()
    monkeypatch.setattr(PygameRenderer, "close", close)
    if after_window:
        monkeypatch.setattr(PygameRenderer, "present", present)
    else:
        monkeypatch.setattr(pygame.key, "get_pressed", fail)
    try:
        with pytest.raises(RuntimeError, match="injected failure"):
            play.main([])
        assert len(closed) == 1
        assert not pygame.display.get_init()
    finally:
        pygame.display.quit()


def test_script_entrypoint_help_works_from_other_directory(tmp_path):
    import subprocess
    import sys
    import play
    result = subprocess.run([sys.executable, play.__file__, "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "--level" in result.stdout
    assert "--scale" in result.stdout


def test_missing_level_reports_cli_error_without_display(tmp_path, capsys):
    import play
    pygame.display.quit()
    assert play.main(["--level", str(tmp_path / "missing.json")]) == 2
    assert "missing.json" in capsys.readouterr().err
    assert not pygame.display.get_init()


def test_main_routes_reset_next_number_and_escape_events(monkeypatch):
    import play
    from rendering import PygameRenderer
    original_present = PygameRenderer.present
    events = [pygame.K_r, pygame.K_n, pygame.K_3, pygame.K_n, pygame.K_ESCAPE]
    frames = []
    def present(renderer, game):
        original_present(renderer, game)
        frames.append((game.level, game.get_state()))
        assert len(frames) <= len(events)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=events[len(frames) - 1]))
    monkeypatch.setattr(PygameRenderer, "present", present)
    assert play.main([]) == 0
    assert [level for level, state in frames] == [
        Level.from_json(play.BUILTIN_LEVELS[i]) for i in (0, 0, 1, 2, 3)]
    assert [state.step_count for level, state in frames] == [1, 1, 1, 1, 1]
    assert not pygame.display.get_init()


def test_terminal_game_stays_frozen_until_explicit_reset(flat_game, event_queue):
    import play
    for _ in range(200):
        flat_game.step(Action.RIGHT)
    assert flat_game.is_won
    terminal = flat_game.get_state()
    game, path, running = play.process_events(flat_game, play.BUILTIN_LEVELS[0], pygame.event.get())
    game.step(Action.RIGHT_JUMP)
    assert game.get_state() == terminal
    assert running
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
    game, _, _ = play.process_events(game, path, pygame.event.get())
    assert not game.is_done
    assert game.step_count == 0
