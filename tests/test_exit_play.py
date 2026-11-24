"""Keyboard/CLI: nine-level catalog, keys 1-9 and cyclic next."""
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

from pathlib import Path

import pygame
import pytest

from game import Game, Level

import play


def test_catalog_discovers_nine_levels_in_name_order():
    assert [p.name for p in play.BUILTIN_LEVELS] == [
        f'level_{i:03}.json' for i in range(1, 10)]


@pytest.mark.parametrize('argv,index', [(["--level", "4"], 3), (["--level", "5"], 4),
                                        (["--level", "6"], 5), (["--level", "7"], 6),
                                        (["--level", "8"], 7), (["--level", "9"], 8)])
def test_cli_accepts_new_level_numbers(monkeypatch, tmp_path, argv, index):
    monkeypatch.chdir(tmp_path)
    args = play.parse_args(argv)
    assert args.level == play.BUILTIN_LEVELS[index]


@pytest.fixture
def flat_game():
    return Game(Level.from_dict(dict(width=240, height=160, player_spawn=[16, 128],
                                     gem=[210, 128], platforms=[[0, 144, 240, 16]])),
                max_steps=None)


@pytest.fixture
def event_queue():
    pygame.display.init()
    pygame.event.clear()
    yield
    pygame.display.quit()


@pytest.mark.parametrize('key,index', [(pygame.K_4, 3), (pygame.K_5, 4), (pygame.K_6, 5),
                                       (pygame.K_7, 6), (pygame.K_8, 7), (pygame.K_9, 8)])
def test_number_keys_4_to_9_load_real_levels(flat_game, event_queue, key, index):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
    game, path, running = play.process_events(flat_game, play.BUILTIN_LEVELS[0],
                                              pygame.event.get())
    assert game is not flat_game
    assert path == play.BUILTIN_LEVELS[index]
    assert game.level == Level.from_json(path)
    assert game.level.win_condition == 'exit_with_gem'
    assert game.max_steps is None and running


def test_next_cycles_from_ninth_level_back_to_first(flat_game, event_queue):
    start = play.BUILTIN_LEVELS[8]
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_n))
    game, path, running = play.process_events(flat_game, start, pygame.event.get())
    assert path == play.BUILTIN_LEVELS[0]
    assert game.level == Level.from_json(path)
    assert running
