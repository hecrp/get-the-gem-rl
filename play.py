"""Human input adapter: the same discrete actions as programmatic play."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pygame

from game import Action, Game
from rendering import PygameRenderer, ViewConfig, ViewMode

def _discover_levels() -> tuple[Path, ...]:
    """All ``level_*.json`` files, in name order, so future levels join the cycle."""
    return tuple(sorted((Path(__file__).resolve().parent / 'levels').glob('level_*.json')))


BUILTIN_LEVELS = _discover_levels()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get the Gem — deterministic pixel platformer")
    parser.add_argument("--level", default="1", help="level number (1-9) or JSON path")
    parser.add_argument("--scale", type=int, default=None,
                        help="integer window scale (default: auto-fit to 1280x800)")
    parser.add_argument("--view", choices=["full", "follow"], default="full",
                        help="view mode: full (default) or follow")
    args = parser.parse_args(argv)
    if args.scale is not None and args.scale < 1:
        parser.error("--scale must be a positive integer")
    if args.level.isdigit() and 1 <= int(args.level) <= len(BUILTIN_LEVELS):
        args.level = BUILTIN_LEVELS[int(args.level) - 1]
    else:
        args.level = Path(args.level).expanduser().resolve()
    return args


def keys_to_action(left: bool, right: bool, jump: bool) -> Action:
    """Opposite directions cancel; holding jump uses the engine's normal semantics."""
    if left == right:
        return Action.JUMP if jump else Action.NOOP
    if left:
        return Action.LEFT_JUMP if jump else Action.LEFT
    return Action.RIGHT_JUMP if jump else Action.RIGHT


def _canvas_size(view: ViewConfig, level) -> tuple[int, int]:
    if view.mode is ViewMode.FULL and view.canvas_size is None:
        return (level.width, level.height)
    if view.mode is ViewMode.FULL:
        return view.canvas_size
    return view.viewport_size


def _auto_scale(canvas_w: int, canvas_h: int) -> int:
    return max(1, min(1280 // canvas_w, 800 // canvas_h))


def process_events(game: Game, level_path: Path, events, renderer=None):
    """Apply discrete episode controls; return the active game, path and running flag."""
    for event in events:
        if event.type == pygame.QUIT:
            return game, level_path, False
        if event.type != pygame.KEYDOWN:
            continue
        if event.key == pygame.K_ESCAPE:
            return game, level_path, False
        if event.key == pygame.K_r:
            game.reset()
        elif pygame.K_1 <= event.key <= pygame.K_9:
            index = event.key - pygame.K_1
            if index < len(BUILTIN_LEVELS):
                level_path = BUILTIN_LEVELS[index]
                game = Game.from_level(level_path, max_steps=None)
        elif event.key == pygame.K_n:
            index = BUILTIN_LEVELS.index(level_path) if level_path in BUILTIN_LEVELS else -1
            level_path = BUILTIN_LEVELS[(index + 1) % len(BUILTIN_LEVELS)]
            game = Game.from_level(level_path, max_steps=None)
        elif event.key == pygame.K_v and renderer is not None:
            new = (ViewMode.FOLLOW if renderer.view.mode is ViewMode.FULL
                   else ViewMode.FULL)
            renderer.set_view(ViewConfig(mode=new, canvas_size=None))
    return game, level_path, True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level_path = args.level
    try:
        game = Game.from_level(level_path, max_steps=None)
    except (OSError, ValueError) as exc:
        print(f"Get the Gem: {exc}", file=sys.stderr)
        return 2
    mode = ViewMode.FULL if args.view == "full" else ViewMode.FOLLOW
    renderer = PygameRenderer(scale=1, view=ViewConfig(mode=mode, canvas_size=None))
    requested_scale = args.scale
    try:
        pygame.display.init()  # event queue needs video, but never audio/font
        clock = pygame.time.Clock()
        while True:
            game, level_path, running = process_events(game, level_path,
                                                       pygame.event.get(), renderer)
            if not running:
                break
            keys = pygame.key.get_pressed()
            action = keys_to_action(keys[pygame.K_LEFT] or keys[pygame.K_a],
                                    keys[pygame.K_RIGHT] or keys[pygame.K_d],
                                    keys[pygame.K_SPACE])
            game.step(action)
            cw, ch = _canvas_size(renderer.view, game.level)
            renderer.scale = requested_scale if requested_scale is not None else _auto_scale(cw, ch)
            renderer.present(game)
            clock.tick(60)
    finally:
        renderer.close()
        pygame.display.quit()  # also closes an event-only display before first present
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
