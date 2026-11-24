"""Whitelist of job flags with type, range and default (spec 13 §97, R-14).

Nothing from a request is ever interpreted by a shell: a validated job becomes
an argument list for ``subprocess.Popen``. Unknown keys, wrong types, values
out of range and strings outside their pattern are rejected (HTTP 422).
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass

NAME_RE = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')
LEVELS_RE = re.compile(r'^[1-9](-[1-9])?(,[1-9](-[1-9])?){0,8}$')
CKPT_RE = re.compile(r'^(model\.pt|ckpt/update_\d{6}\.pt)$')
PRESETS = ('tiny', 'small', 'wide', 'tall', 'large', 'huge')
DEVICES = ('auto', 'cpu', 'mps', 'cuda')
CORES = os.cpu_count() or 4


class JobError(ValueError):
    def __init__(self, detail: str, status: int = 422, code: str = 'invalid_job'):
        super().__init__(detail)
        self.detail, self.status, self.code = detail, status, code


@dataclass(frozen=True)
class Flag:
    kind: str                     # int | float | bool | str | enum
    lo: float | None = None
    hi: float | None = None
    choices: tuple = ()
    pattern: re.Pattern | None = None
    cli: str | None = None        # override of the --flag spelling


def valid_run_name(name) -> bool:
    return (isinstance(name, str) and bool(NAME_RE.match(name)) and '..' not in name
            and not name.startswith(('_', '.', '-')))


I = lambda lo, hi: Flag('int', lo, hi)          # noqa: E731
F = lambda lo, hi: Flag('float', lo, hi)        # noqa: E731

TRAIN = {
    'levels': Flag('str', pattern=LEVELS_RE),
    'preset': Flag('enum', choices=PRESETS),
    'start_seed': I(0, 2 ** 62), 'num_levels': I(0, 10_000_000),
    'frame_skip': I(1, 16), 'gem_reward': F(-10, 10), 'death_penalty': F(0, 10),
    'step_penalty': F(0, 1), 'distance_shaping': F(0, 10), 'no_goal_vector': Flag('bool'),
    'total_steps': I(1, 200_000_000), 'num_envs': I(1, 256), 'num_steps': I(8, 4096),
    'lr': F(1e-7, 1), 'gamma': F(0.5, 1), 'gae_lambda': F(0, 1), 'minibatches': I(1, 256),
    'epochs': I(1, 50), 'clip': F(0.01, 1), 'ent_coef': F(0, 1), 'vf_coef': F(0, 10),
    'max_grad_norm': F(0.01, 100), 'seed': I(0, 2 ** 31 - 1),
    'name': Flag('str', pattern=NAME_RE), 'group': Flag('str', pattern=NAME_RE),
    'threads': I(1, CORES), 'device': Flag('enum', choices=DEVICES),
    'log_every': I(1, 100_000), 'eval_every': I(0, 1_000_000), 'eval_episodes': I(1, 10_000),
    'eval_envs': I(1, 256), 'eval_levels': I(1, 1_000_000),
    'ckpt_every': I(0, 1_000_000), 'ckpt_keep': I(0, 10_000), 'ckpt_thin': I(0, 1_000_000),
    'replay_every': I(0, 1_000_000), 'replay_n': I(0, 64),
    'resume': Flag('str', pattern=re.compile(r'^(latest|ckpt/update_\d{6}\.pt)$')),
}
ROLLOUT = {
    'run': Flag('str', pattern=NAME_RE), 'ckpt': Flag('str', pattern=CKPT_RE),
    'levels': Flag('str', pattern=LEVELS_RE), 'preset': Flag('enum', choices=PRESETS),
    'start_seed': I(0, 2 ** 62), 'num': I(1, 20_000), 'repeat': I(1, 1000),
    'policy': Flag('enum', choices=('greedy', 'sampled')),
    'tag': Flag('str', pattern=NAME_RE), 'seed': I(0, 2 ** 31 - 1),
    'keep_replays': I(0, 64), 'workers': I(1, CORES), 'envs': I(1, 256),
}
SWEEP = dict(ROLLOUT, max_episodes=I(1, 20_000))
SCHEMAS = {'train': TRAIN, 'rollout': ROLLOUT, 'sweep': SWEEP}
MODULES = {'train': 'rl.ppo', 'rollout': 'rl.rollout', 'sweep': 'rl.rollout'}


def _value(key: str, flag: Flag, value):
    if flag.kind == 'bool':
        if not isinstance(value, bool):
            raise JobError(f'{key}: expected true or false')
        return value
    if flag.kind == 'int':
        if isinstance(value, bool) or not isinstance(value, int):
            raise JobError(f'{key}: expected an integer')
    elif flag.kind == 'float':
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isfinite(value):
            raise JobError(f'{key}: expected a finite number')
        value = float(value)
    elif flag.kind == 'enum':
        if value not in flag.choices:
            raise JobError(f'{key}: expected one of {", ".join(flag.choices)}')
        return value
    elif flag.kind == 'str':
        if not isinstance(value, str) or not flag.pattern.match(value):
            raise JobError(f'{key}: invalid value')
        if key in ('name', 'run') and not valid_run_name(value):
            raise JobError(f'{key}: invalid run name (A-Z a-z 0-9 _ . -, not starting '
                           f'with "_", no "..")')
        return value
    if flag.lo is not None and value < flag.lo or flag.hi is not None and value > flag.hi:
        raise JobError(f'{key}: out of range [{flag.lo:g}, {flag.hi:g}]')
    return value


def validate(kind: str, args) -> dict:
    """Checked copy of ``args`` (unknown keys -> ``JobError``)."""
    if kind not in SCHEMAS:
        raise JobError(f'kind: expected one of {", ".join(SCHEMAS)}')
    if not isinstance(args, dict):
        raise JobError('args: expected an object')
    schema = SCHEMAS[kind]
    out = {}
    for key, value in args.items():
        if key not in schema:
            raise JobError(f'{key}: flag not allowed for {kind} jobs')
        out[key] = _value(key, schema[key], value)
    if kind == 'train':
        if 'preset' in out and 'levels' in out:
            raise JobError('levels and preset are exclusive')
    else:
        if 'run' not in out:
            raise JobError('run: required')
        if 'preset' in out and 'levels' in out:
            raise JobError('levels and preset are exclusive')
    return out


def run_name(kind: str, args: dict) -> str:
    """The run a job writes to (as ``rl.ppo`` would name it)."""
    if kind != 'train':
        return args['run']
    if args.get('name'):
        return args['name']
    preset = args.get('preset')
    levels = args.get('levels', '1-6')
    n = args.get('num_levels', 200) if preset else ''
    return f'{preset or "levels" + levels}_n{n}_s{args.get("seed", 1)}'


def argv(kind: str, args: dict, python: str, runs_dir: str) -> list[str]:
    """The argument list (never a shell string) for a validated job."""
    out = [python, '-m', MODULES[kind]]
    if kind == 'train':
        args = dict(args)
        args.setdefault('name', run_name(kind, args))
        for key, value in args.items():
            flag = '--' + key.replace('_', '-')
            if key == 'resume':
                out.append('--resume')
                if value != 'latest':
                    out.append(value)
            elif isinstance(value, bool):
                if value:
                    out.append(flag)
            else:
                out += [flag, repr(value) if isinstance(value, float) else str(value)]
        out += ['--runs-dir', runs_dir]
        return out
    out += ['--mode', 'sweep' if kind == 'sweep' else 'replays']
    for key, value in args.items():
        if key == 'policy':
            out.append('--' + value)
            continue
        out += ['--' + key.replace('_', '-'), str(value)]
    out += ['--runs-dir', runs_dir]
    return out


def describe() -> dict:
    """JSON description of the whitelist (for the Operate forms)."""
    def one(f: Flag):
        d = {'type': f.kind}
        if f.lo is not None:
            d['min'], d['max'] = f.lo, f.hi
        if f.choices:
            d['choices'] = list(f.choices)
        if f.pattern is not None:
            d['pattern'] = f.pattern.pattern
        return d
    return {k: {name: one(f) for name, f in s.items()} for k, s in SCHEMAS.items()}
