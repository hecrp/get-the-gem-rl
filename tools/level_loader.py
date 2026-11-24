"""Validate a level: python -m tools.level_loader levels/level_001.json."""
import argparse
import sys
from game import Level


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Validate a Get the Gem JSON level')
    parser.add_argument('path', help='Path to the JSON level')
    args = parser.parse_args(argv)
    try:
        level = Level.from_json(args.path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f'OK {args.path}: {level.name or "unnamed"}, '
          f'{level.width}x{level.height}, {len(level.platforms)} platforms, '
          f'{len(level.hazards)} hazards')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
