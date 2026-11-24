"""Record numbered replays or evaluation sweeps from a trained checkpoint.

    python -m rl.rollout --run tiny_n100_shaped_s1 --preset tiny --start-seed 1000000 --num 16
    python -m rl.rollout --run levels1-6_s1 --levels 1-6 --sampled --ckpt ckpt/update_000300.pt
    python -m rl.rollout --run tiny_n10_shaped_s1 --mode sweep --tag test1k \\
        --preset tiny --start-seed 1000000 --num 1000

A replay is ``level + actions`` (spec 13 §91.3): the episode can be rebuilt
exactly with ``game.replay.simulate``, which ``--verify`` (on by default in
``replays`` mode) checks for every episode. Several ``GemEnv`` run side by side
with one batched network call per step and no autoreset; the level of each
episode is fixed by the caller, so a rollout is reproducible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from game import Level
from game.replay import REPLAY_SCHEMA, rle_encode, simulate
from game.replay_store import append_jsonl, append_replays, atomic_write_json, store_level

from .env import BUILTIN_LEVELS, GRID_CHANNELS, VEC_FIELDS, GemEnv

ROOT = Path(__file__).resolve().parents[1]
TEST_START_SEED = 1_000_000
GRID_SHAPE = (len(GRID_CHANNELS), 21, 31)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# -- policy ----------------------------------------------------------------------

@dataclass
class Policy:
    agent: object
    config: dict
    path: str
    meta: dict = field(default_factory=dict)      # update / env_steps of a checkpoint

    @property
    def frame_skip(self) -> int:
        return int(self.config.get('frame_skip', 4))

    def env_kwargs(self) -> dict:
        c = self.config
        return dict(frame_skip=self.frame_skip,
                    gem_reward=float(c.get('gem_reward', 0.0)),
                    death_penalty=float(c.get('death_penalty', 0.0)),
                    step_penalty=float(c.get('step_penalty', 0.0)),
                    distance_shaping=float(c.get('distance_shaping', 0.0)),
                    goal_vector=not c.get('no_goal_vector', False))


def build_agent(device='cpu'):
    from .ppo import Agent
    return Agent(GRID_SHAPE, len(VEC_FIELDS), 6).to(device)


def load_policy(path, device='cpu') -> Policy:
    """Rebuild the PPO ``Agent`` from ``model.pt`` or a ``ckpt/update_*.pt``.

    The architecture comes from ``GRID_CHANNELS``/``VEC_FIELDS`` (as in
    training) and ``goal_vector``/``frame_skip``/rewards from the run config
    stored in the checkpoint. A mismatching checkpoint fails with a clear
    message instead of producing garbage actions.
    """
    path = Path(path)
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f'{path}: cannot load checkpoint: {exc}') from exc
    if not isinstance(ckpt, dict) or 'model' not in ckpt:
        raise ValueError(f'{path}: not a GemEnv PPO checkpoint (no "model")')
    agent = build_agent(device)
    try:
        agent.load_state_dict(ckpt['model'])
    except RuntimeError as exc:
        raise ValueError(
            f'{path}: checkpoint does not match the current observation layout '
            f'(grid {GRID_SHAPE}, vec {len(VEC_FIELDS)}): {exc}') from exc
    agent.eval()
    meta = {k: ckpt[k] for k in ('update', 'env_steps') if k in ckpt}
    return Policy(agent, dict(ckpt.get('config') or {}), str(path), meta)


# -- level references ------------------------------------------------------------

def parse_levels(text: str) -> list[int]:
    out = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def builtin_ref(n: int) -> dict:
    if not 1 <= n <= len(BUILTIN_LEVELS):
        raise ValueError(f'level: built-in levels are 1..{len(BUILTIN_LEVELS)}')
    return {'kind': 'builtin', 'n': n}


def preset_refs(preset: str, start_seed: int, num: int) -> list[dict]:
    from pcg import PRESETS, pcg_ref
    if preset not in PRESETS:
        raise ValueError(f'preset: expected one of {", ".join(PRESETS)}')
    return [pcg_ref(s, PRESETS[preset], preset) for s in range(start_seed, start_seed + num)]


def resolve_ref(ref: dict) -> Level:
    kind = ref.get('kind')
    if kind == 'builtin':
        return Level.from_json(BUILTIN_LEVELS[ref['n'] - 1])
    if kind == 'pcg':
        from pcg import GenConfig, generate_level
        return generate_level(ref['seed'], GenConfig(**ref['config']))
    raise ValueError(f'level_ref: cannot resolve {ref!r}')


def split_of(ref: dict, config: dict) -> str:
    """``train``/``test`` relative to the run's seed ranges; ``builtin`` for
    built-in levels; ``other`` otherwise."""
    if ref.get('kind') == 'builtin':
        return 'builtin'
    if ref.get('kind') != 'pcg':
        return 'other'
    seed = ref['seed']
    if config.get('preset') and ref.get('preset') == config.get('preset'):
        start = int(config.get('start_seed', 0))
        num = int(config.get('num_levels', 0))
        if num == 0 and seed < TEST_START_SEED:
            return 'train'
        if start <= seed < start + num:
            return 'train'
        if seed >= TEST_START_SEED:
            return 'test'
    return 'other'


# -- running episodes --------------------------------------------------------------

@dataclass
class Episode:
    index: int
    ref: dict
    level: Level
    actions: list[int]
    ret: float
    info: dict
    max_steps: object
    won: bool
    dead: bool
    timeout: bool
    final_xy: tuple[float, float]
    final_state: object


@torch.no_grad()
def run_episodes(policy: Policy, refs: list[dict], *, greedy: bool, seed: int = 0,
                 n_envs: int = 16, device='cpu', levels: list[Level] | None = None,
                 progress=None) -> list[Episode]:
    """Play one episode per ``refs[i]`` with up to ``n_envs`` envs in lock-step.

    Sampling uses a private ``torch.Generator`` seeded with ``seed`` (never
    the global RNG), so recording during training does not perturb it.
    """
    levels = levels if levels is not None else [resolve_ref(r) for r in refs]
    n_envs = max(1, min(n_envs, len(refs)))
    gen = torch.Generator(device='cpu')
    gen.manual_seed(seed)
    envs = [GemEnv(levels=levels[0], **policy.env_kwargs()) for _ in range(n_envs)]
    slots: list[dict | None] = [None] * n_envs
    obs: list[dict | None] = [None] * n_envs
    nxt = 0
    out: list[Episode | None] = [None] * len(refs)

    def start(slot):
        nonlocal nxt
        if nxt >= len(refs):
            slots[slot] = None
            return
        i = nxt
        nxt += 1
        o, _ = envs[slot].reset(seed=seed + i, options={'level': levels[i]})
        obs[slot] = o
        slots[slot] = {'i': i, 'actions': [], 'ret': 0.0}

    for s in range(n_envs):
        start(s)
    agent = policy.agent
    while any(slots):
        active = [s for s in range(n_envs) if slots[s] is not None]
        grid = torch.as_tensor(np.stack([obs[s]['grid'] for s in active]), device=device)
        vec = torch.as_tensor(np.stack([obs[s]['vec'] for s in active]), device=device)
        logits = agent.actor(agent._features(grid, vec))
        if greedy:
            acts = logits.argmax(-1).cpu()
        else:
            probs = torch.softmax(logits.float(), -1).cpu()
            acts = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        for s, a in zip(active, acts.tolist()):
            st = slots[s]
            o, r, term, trunc, info = envs[s].step(a)
            st['actions'].append(a)
            st['ret'] += float(r)
            obs[s] = o
            if term or trunc:
                g = envs[s].game
                i = st['i']
                out[i] = Episode(i, refs[i], levels[i], st['actions'], st['ret'], info,
                                 g.max_steps, g.is_won, g.is_dead, g.is_timeout,
                                 (g.player.x, g.player.y), g.get_state())
                if progress:
                    progress(i)
                start(s)
    for e in envs:
        e.close()
    return out


def _worker(job):
    path, config, refs, greedy, seed, n_envs, offset = job
    torch.set_num_threads(1)
    policy = load_policy(path)
    policy.config = config
    eps = run_episodes(policy, refs, greedy=greedy, seed=seed, n_envs=n_envs)
    for ep in eps:
        ep.index += offset
    return eps


def play(policy: Policy, refs: list[dict], *, greedy: bool, seed: int = 0,
         n_envs: int = 16, workers: int = 1) -> list[Episode]:
    """``run_episodes`` in this process, or split in ``workers`` contiguous
    chunks over a process pool (CPU inference does not scale with threads;
    the convolution is the bottleneck). Chunk ``k`` samples with seed
    ``seed + k``, so a result is reproducible for a given ``workers``."""
    if workers <= 1 or len(refs) < 2 * workers:
        return run_episodes(policy, refs, greedy=greedy, seed=seed, n_envs=n_envs)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    size = -(-len(refs) // workers)
    jobs = [(policy.path, policy.config, refs[i:i + size], greedy, seed + k,
             max(1, n_envs // workers), i)
            for k, i in enumerate(range(0, len(refs), size))]
    with ProcessPoolExecutor(len(jobs), mp_context=mp.get_context('spawn')) as pool:
        chunks = list(pool.map(_worker, jobs))
    return [ep for chunk in chunks for ep in chunk]


def verify(ep: Episode, frame_skip: int):
    """Re-simulate with ``game.replay`` and compare with what ``GemEnv`` saw.
    Returns the trajectory summary; raises ``AssertionError`` on mismatch."""
    traj = simulate(ep.level, rle_encode(ep.actions), frame_skip=frame_skip,
                    max_steps=ep.max_steps, record=False)
    expected = (ep.won, ep.dead, ep.timeout, ep.info['step_count'], ep.final_state)
    got = (traj.result == 'won', traj.result == 'dead', traj.result == 'timeout',
           traj.engine_steps, traj.final_state)
    if expected != got:
        raise AssertionError(
            f'episode {ep.index}: replay mismatch: env (won, dead, timeout, steps) = '
            f'{expected[:4]}, simulate = {got[:4]}')
    return traj


# -- records ------------------------------------------------------------------------

def source_of(policy: Policy, run_dir: Path, kind: str | None = None) -> dict:
    path = Path(policy.path)
    try:
        rel = str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        rel = str(path)
    if kind is None:
        kind = 'checkpoint' if rel.startswith('ckpt') else 'final'
    src = {'kind': kind, 'ckpt': rel}
    if 'update' in policy.meta:
        src['update'] = int(policy.meta['update'])
    if 'env_steps' in policy.meta:
        src['env_steps'] = int(policy.meta['env_steps'])
    return src


def replay_record(ep: Episode, traj, *, policy_name: str, source: dict, split: str,
                  frame_skip: int, digest: str, tags=()) -> dict:
    outcome = traj.outcome()
    outcome['return'] = round(ep.ret, 6)
    return {
        'schema': REPLAY_SCHEMA, 'created': now_iso(), 'source': source,
        'policy': policy_name, 'split': split,
        'level': {'ref': ep.ref, 'hash': digest, 'name': ep.level.name,
                  'width': ep.level.width, 'height': ep.level.height},
        'env': {'frame_skip': frame_skip, 'max_steps': ep.max_steps, 'physics': None},
        'actions': rle_encode(ep.actions), 'outcome': outcome, 'tags': list(tags),
    }


def level_metrics(level: Level, ref: dict) -> dict:
    out = {'width': level.width, 'height': level.height}
    if ref.get('kind') == 'pcg' or level._metadata.get('generator'):
        from pcg import metrics
        m = metrics(level)
        out.update(min_window=m.min_window, mean_window=m.mean_window,
                   linearity=round(m.linearity, 4), leniency=round(m.leniency, 4),
                   density=round(m.density, 4), path_ledges=m.path_ledges)
    return out


def record_replays(policy: Policy, refs: list[dict], run_dir, runs_dir, *, greedy: bool,
                   seed: int = 0, n_envs: int = 16, check: bool = True, tags=(),
                   source: dict | None = None, corrupt: bool = False,
                   workers: int = 1) -> list[dict]:
    """Play, verify and append replays (numbered under lock). Library entry
    point, also used by ``rl.ppo --replay-every``."""
    run_dir = Path(run_dir)
    eps = play(policy, refs, greedy=greedy, seed=seed, n_envs=n_envs, workers=workers)
    source = source or source_of(policy, run_dir)
    fs = policy.frame_skip
    records = []
    for ep in eps:
        if corrupt:                               # test hook for --verify
            ep.actions[0] = (ep.actions[0] + 1) % 6
        traj = verify(ep, fs) if check else simulate(
            ep.level, rle_encode(ep.actions), frame_skip=fs, max_steps=ep.max_steps,
            record=False)
        digest = store_level(runs_dir, ep.level)
        records.append(replay_record(ep, traj, policy_name='greedy' if greedy else 'sampled',
                                     source=source, split=split_of(ep.ref, policy.config),
                                     frame_skip=fs, digest=digest, tags=tags))
    return append_replays(run_dir, records)


def sweep(policy: Policy, refs: list[dict], run_dir, runs_dir, *, greedy: bool, tag: str,
          seed: int = 0, n_envs: int = 16, keep: int = 4, check: bool = False,
          workers: int = 1) -> dict:
    run_dir = Path(run_dir)
    t0 = time.perf_counter()
    eps = play(policy, refs, greedy=greedy, seed=seed, n_envs=n_envs, workers=workers)
    t_play = time.perf_counter() - t0
    fs = policy.frame_skip
    name = 'greedy' if greedy else 'sampled'
    lines, trajs = [], []
    for ep in eps:
        traj = verify(ep, fs) if check else simulate(
            ep.level, rle_encode(ep.actions), frame_skip=fs, max_steps=ep.max_steps,
            record=False)
        trajs.append(traj)
        hist = np.bincount(np.asarray(ep.actions, dtype=np.int64), minlength=6)
        from game.level import level_hash
        lines.append({
            'schema': 1, 'i': ep.index, 'level_ref': ep.ref, 'level_hash': level_hash(ep.level),
            'policy': name, 'success': traj.result == 'won',
            'gem': traj.gem_step is not None, 'engine_steps': traj.engine_steps,
            'env_steps': traj.env_steps, 'end_reason': traj.end_reason,
            'gem_step': traj.gem_step, 'win_step': traj.win_step,
            'death_xy': None if traj.death_xy is None else list(traj.death_xy),
            'level_metrics': level_metrics(ep.level, ep.ref),
            'action_hist': hist.tolist(), 'return': round(ep.ret, 6),
            'split': split_of(ep.ref, policy.config), 'frame_skip': fs,
            # Not in the spec's minimal line: lets the server rebuild visit
            # heat maps by re-simulation (actions are a few hundred bytes).
            'actions': rle_encode(ep.actions),
        })
    sweeps = run_dir / 'sweeps'
    path = sweeps / f'{tag}.jsonl'
    if path.exists():
        path.unlink()
    append_jsonl(path, lines)
    # Full replays: the first ``keep`` episodes plus the best (shortest win)
    # and the worst (longest failure, else longest episode).
    chosen = list(range(min(keep, len(eps))))
    wins = [i for i, t in enumerate(trajs) if t.result == 'won']
    if wins:
        chosen.append(min(wins, key=lambda i: trajs[i].engine_steps))
    fails = [i for i, t in enumerate(trajs) if t.result != 'won']
    pool = fails or list(range(len(eps)))
    if pool:
        chosen.append(max(pool, key=lambda i: trajs[i].engine_steps))
    source = source_of(policy, run_dir)
    source['sweep'] = tag
    recs = []
    for i in dict.fromkeys(chosen):
        ep = eps[i]
        digest = store_level(runs_dir, ep.level)
        tags = [f'sweep:{tag}']
        if wins and i == min(wins, key=lambda j: trajs[j].engine_steps):
            tags.append('best')
        if pool and i == max(pool, key=lambda j: trajs[j].engine_steps):
            tags.append('worst')
        recs.append(replay_record(ep, trajs[i], policy_name=name, source=source,
                                  split=split_of(ep.ref, policy.config), frame_skip=fs,
                                  digest=digest, tags=tags))
    replays = append_replays(run_dir, recs)
    from pcg import GENERATOR_VERSION
    seeds = [r['seed'] for r in refs if r.get('kind') == 'pcg']
    meta = {'schema': 1, 'tag': tag, 'ckpt': source.get('ckpt'), 'source': source,
            'policy': name, 'n': len(lines), 'created': now_iso(),
            'generator_version': GENERATOR_VERSION,
            'seed_range': [min(seeds), max(seeds)] if seeds else None,
            'levels': sorted({r['n'] for r in refs if r.get('kind') == 'builtin'}) or None,
            'seed': seed, 'workers': workers, 'play_seconds': round(t_play, 3),
            'env_steps': int(sum(t.env_steps for t in trajs)),
            'replays': [r['id'] for r in replays]}
    atomic_write_json(sweeps / f'{tag}.meta.json', meta)
    return meta


# -- CLI ------------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run', required=True, help='run name under --runs-dir')
    ap.add_argument('--runs-dir', default=str(ROOT / 'runs'))
    ap.add_argument('--ckpt', default='model.pt',
                    help='checkpoint relative to the run (model.pt or ckpt/update_N.pt)')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--levels', default=None, help='built-in levels, e.g. 1-6')
    src.add_argument('--preset', default=None, help='pcg preset')
    ap.add_argument('--start-seed', type=int, default=TEST_START_SEED)
    ap.add_argument('--num', type=int, default=None,
                    help='episodes (preset: seeds start..start+num-1; levels: cycles the list)')
    pol = ap.add_mutually_exclusive_group()
    pol.add_argument('--greedy', action='store_true', help='argmax actions (default)')
    pol.add_argument('--sampled', action='store_true', help='sample from the policy')
    ap.add_argument('--tag', default=None, help='replay tag / sweep name')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mode', default='replays', choices=('replays', 'sweep'))
    ap.add_argument('--keep-replays', type=int, default=4,
                    help='sweep: full replays kept (first K, plus best and worst)')
    ap.add_argument('--repeat', type=int, default=1,
                    help='episodes per level (useful with --sampled)')
    ap.add_argument('--max-episodes', type=int, default=None, help='cap on episodes')
    ap.add_argument('--envs', type=int, default=16, help='parallel environments')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--workers', type=int, default=1,
                    help='worker processes (CPU inference scales with processes, not threads)')
    ap.add_argument('--verify', dest='verify', action='store_true', default=None,
                    help='re-simulate every episode (default in replays mode)')
    ap.add_argument('--no-verify', dest='verify', action='store_false')
    ap.add_argument('--debug-corrupt', action='store_true', help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def refs_from_args(args, config: dict) -> list[dict]:
    if args.preset is None and args.levels is None:
        # Default: the run's own setting, on held-out seeds for pcg runs.
        if config.get('preset'):
            args.preset = config['preset']
        else:
            args.levels = config.get('levels', '1-9')
    if args.preset is not None:
        refs = preset_refs(args.preset, args.start_seed, args.num if args.num is not None else 16)
    else:
        nums = parse_levels(args.levels)
        refs = [builtin_ref(n) for n in nums]
        if args.num is not None:
            refs = [refs[i % len(refs)] for i in range(args.num)]
    if args.repeat > 1:
        refs = [r for r in refs for _ in range(args.repeat)]
    if args.max_episodes is not None:
        refs = refs[:args.max_episodes]
    if not refs:
        raise ValueError('no episodes to play')
    return refs


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        torch.set_num_threads(max(1, args.threads))
        runs_dir = Path(args.runs_dir)
        run_dir = runs_dir / args.run
        if not run_dir.is_dir():
            raise ValueError(f'run not found: {run_dir}')
        ckpt = Path(args.ckpt)
        ckpt = ckpt if ckpt.is_absolute() else run_dir / ckpt
        policy = load_policy(ckpt, args.device)
        config = policy.config or json.loads((run_dir / 'config.json').read_text())
        policy.config = config
        refs = refs_from_args(args, config)
        greedy = not args.sampled
        start = time.perf_counter()
        if args.mode == 'replays':
            check = True if args.verify is None else args.verify
            tags = [args.tag] if args.tag else []
            recs = record_replays(policy, refs, run_dir, runs_dir, greedy=greedy,
                                  seed=args.seed, n_envs=args.envs, check=check, tags=tags,
                                  corrupt=args.debug_corrupt, workers=args.workers)
            secs = time.perf_counter() - start
            steps = sum(r['outcome']['env_steps'] for r in recs)
            summary = {'mode': 'replays', 'run': args.run, 'replays': len(recs),
                       'ids': [r['id'] for r in recs],
                       'path': str(run_dir / 'replays.jsonl'),
                       'success_rate': sum(r['outcome']['result'] == 'won' for r in recs) / len(recs),
                       'seconds': round(secs, 3), 'env_steps_per_s': round(steps / secs)}
        else:
            tag = args.tag or f'sweep_{int(time.time())}'
            check = bool(args.verify)
            meta = sweep(policy, refs, run_dir, runs_dir, greedy=greedy, tag=tag,
                         seed=args.seed, n_envs=args.envs, keep=args.keep_replays,
                         check=check, workers=args.workers)
            secs = time.perf_counter() - start
            lines = [json.loads(x) for x in
                     (run_dir / 'sweeps' / f'{tag}.jsonl').read_text().splitlines()]
            summary = {'mode': 'sweep', 'run': args.run, 'tag': tag, 'episodes': meta['n'],
                       'path': str(run_dir / 'sweeps' / f'{tag}.jsonl'),
                       'success_rate': sum(x['success'] for x in lines) / len(lines),
                       'replays': meta['replays'], 'seconds': round(secs, 3),
                       'env_steps_per_s': round(meta['env_steps'] / meta['play_seconds'])}
        print(json.dumps(summary), flush=True)
        return 0
    except (ValueError, AssertionError, OSError, KeyError) as exc:
        print(f'rl.rollout: error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
