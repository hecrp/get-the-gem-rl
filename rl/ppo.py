"""PPO baseline for GemEnv (single file, CleanRL style).

A small CNN reads the egocentric tile grid, an MLP reads the feature vector,
and both feed a shared actor-critic head. Truncated episodes (timeouts) are
bootstrapped from the value of their final observation.

    # built-in levels 1-6, gem pickup bonus
    python -m rl.ppo --levels 1-6 --gem-reward 0.5 --total-steps 2000000

    # generalisation: train on 200 generated levels, evaluate on unseen seeds
    python -m rl.ppo --preset tiny --num-levels 200 --total-steps 5000000

Writes ``runs/<name>/{config.json, progress.csv, eval.json, model.pt}``, plus
``status.json`` (state, heartbeat, progress) and ``metrics.jsonl`` (the
``progress.csv`` columns and a few more), which older readers ignore. Opt-in:
``--ckpt-every`` (resumable checkpoints), ``--replay-every`` (numbered replays
of the current policy, see ``rl.rollout``), ``--resume`` and ``--group``.
SIGTERM/SIGINT stop cleanly after the current update: checkpoint, ``model.pt``,
a partial ``eval.json`` and ``state: "stopped"``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from game.replay_store import append_jsonl, atomic_write_json

from .env import GRID_CHANNELS, VEC_FIELDS, GemEnv

ROOT = Path(__file__).resolve().parents[1]
TEST_START_SEED = 1_000_000          # [0, 1e6) train, [1e6, 2e6) test


# -- model ---------------------------------------------------------------------

def _init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    def __init__(self, grid_shape, vec_dim, n_actions, hidden=256):
        super().__init__()
        c, h, w = grid_shape
        self.cnn = nn.Sequential(
            _init(nn.Conv2d(c, 32, 3, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.ReLU(),
            _init(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.ReLU(),
            nn.Flatten())
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.vec = nn.Sequential(_init(nn.Linear(vec_dim, 64)), nn.ReLU())
        self.body = nn.Sequential(_init(nn.Linear(n_flat + 64, hidden)), nn.ReLU())
        self.actor = _init(nn.Linear(hidden, n_actions), std=0.01)
        self.critic = _init(nn.Linear(hidden, 1), std=1.0)

    def _features(self, grid, vec):
        return self.body(torch.cat([self.cnn(grid.float()), self.vec(vec)], dim=1))

    def value(self, grid, vec):
        return self.critic(self._features(grid, vec)).squeeze(-1)

    def act(self, grid, vec, action=None, greedy=False):
        z = self._features(grid, vec)
        dist = Categorical(logits=self.actor(z))
        if action is None:
            action = dist.probs.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(z).squeeze(-1)


# -- environments --------------------------------------------------------------

def _parse_levels(text):
    out = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def level_source(args, test=False):
    """Built-in level list, or a pcg LevelDistribution (train or test seeds)."""
    if args.preset is None:
        return _parse_levels(args.levels)
    from pcg import PRESETS, LevelDistribution
    if test:
        return LevelDistribution(PRESETS[args.preset], TEST_START_SEED, args.eval_levels)
    return LevelDistribution(PRESETS[args.preset], args.start_seed, args.num_levels)


def make_vec_env(levels, n, args):
    def thunk():
        return GemEnv(levels=levels, frame_skip=args.frame_skip,
                      gem_reward=args.gem_reward, death_penalty=args.death_penalty,
                      step_penalty=args.step_penalty, goal_vector=not args.no_goal_vector,
                      distance_shaping=args.distance_shaping)
    return gym.vector.SyncVectorEnv([thunk for _ in range(n)],
                                    autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def to_tensors(obs, device):
    return (torch.as_tensor(obs['grid'], device=device),
            torch.as_tensor(obs['vec'], device=device))


def final_obs_at(final_obs, i):
    """The final observation of sub-env ``i`` (Gymnasium may batch it as a dict
    of arrays or as an object array of dicts)."""
    if isinstance(final_obs, dict):
        return {k: v[i] for k, v in final_obs.items()}
    return final_obs[i]


# -- evaluation ----------------------------------------------------------------

@torch.no_grad()
def evaluate(agent, levels, episodes, args, device, greedy=False, seed=12345):
    """Success rate over ``episodes`` episodes on ``levels`` (vectorised)."""
    n = min(args.eval_envs, episodes)
    envs = make_vec_env(levels, n, args)
    obs, _ = envs.reset(seed=seed)
    wins, gems, returns = [], [], []
    ep_ret = np.zeros(n)
    while len(wins) < episodes:
        g, v = to_tensors(obs, device)
        action, *_ = agent.act(g, v, greedy=greedy)
        obs, reward, term, trunc, info = envs.step(action.cpu().numpy())
        ep_ret += reward
        for i in np.flatnonzero(term | trunc):
            fi = info['final_info']
            wins.append(bool(fi['is_success'][i]))
            gems.append(bool(fi['has_gem'][i]))
            returns.append(ep_ret[i])
            ep_ret[i] = 0.0
    envs.close()
    wins, gems, returns = wins[:episodes], gems[:episodes], returns[:episodes]
    return {'episodes': episodes, 'success_rate': float(np.mean(wins)),
            'gem_rate': float(np.mean(gems)), 'mean_return': float(np.mean(returns))}


def evaluate_all(agent, args, device):
    """Per built-in level, or train seeds vs held-out test seeds for pcg."""
    out = {}
    for greedy in (False, True):
        key = 'greedy' if greedy else 'sampled'
        if args.preset is None:
            out[key] = {str(n): evaluate(agent, [n], args.eval_episodes, args, device, greedy)
                        for n in _parse_levels(args.levels)}
        else:
            out[key] = {
                'train': evaluate(agent, level_source(args), args.eval_episodes, args,
                                  device, greedy),
                'test': evaluate(agent, level_source(args, test=True), args.eval_episodes,
                                 args, device, greedy),
            }
    return out


# -- training ------------------------------------------------------------------

def compute_gae(rewards, dones, values, next_value, gamma, lam):
    """Generalised advantage estimates, shape (T, N).

    ``dones[t]`` marks that the episode ended *at* step t (Gymnasium same-step
    autoreset), so the value that follows belongs to a new episode and is not
    bootstrapped. Time-limit bootstrapping is already folded into ``rewards``.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_argument_group('levels')
    src.add_argument('--levels', default='1-6', help='built-in levels, e.g. 1-6')
    src.add_argument('--preset', default=None, help='pcg preset; overrides --levels')
    src.add_argument('--start-seed', type=int, default=0)
    src.add_argument('--num-levels', type=int, default=200, help='0 = unbounded')
    env = ap.add_argument_group('environment')
    env.add_argument('--frame-skip', type=int, default=4)
    env.add_argument('--gem-reward', type=float, default=0.5)
    env.add_argument('--death-penalty', type=float, default=0.0)
    env.add_argument('--step-penalty', type=float, default=0.0)
    env.add_argument('--distance-shaping', type=float, default=0.0,
                     help='potential-based shaping coefficient (0 = off)')
    env.add_argument('--no-goal-vector', action='store_true')
    ppo = ap.add_argument_group('ppo')
    ppo.add_argument('--total-steps', type=int, default=2_000_000, help='env steps')
    ppo.add_argument('--num-envs', type=int, default=32)
    ppo.add_argument('--num-steps', type=int, default=128, help='rollout length')
    ppo.add_argument('--lr', type=float, default=5e-4)
    ppo.add_argument('--gamma', type=float, default=0.995)
    ppo.add_argument('--gae-lambda', type=float, default=0.95)
    ppo.add_argument('--minibatches', type=int, default=8)
    ppo.add_argument('--epochs', type=int, default=3)
    ppo.add_argument('--clip', type=float, default=0.2)
    ppo.add_argument('--ent-coef', type=float, default=0.01)
    ppo.add_argument('--vf-coef', type=float, default=0.5)
    ppo.add_argument('--max-grad-norm', type=float, default=0.5)
    run = ap.add_argument_group('run')
    run.add_argument('--seed', type=int, default=1)
    run.add_argument('--name', default=None)
    run.add_argument('--threads', type=int, default=4)
    run.add_argument('--device', default='auto', help='auto, cpu, mps or cuda')
    run.add_argument('--log-every', type=int, default=10, help='updates')
    run.add_argument('--eval-every', type=int, default=0,
                     help='updates between evaluations (0 = only at the end)')
    run.add_argument('--eval-episodes', type=int, default=64)
    run.add_argument('--eval-envs', type=int, default=16)
    run.add_argument('--eval-levels', type=int, default=1000, help='held-out pcg seeds')
    run.add_argument('--runs-dir', default=None, help='parent of the run folder (default: runs/)')
    run.add_argument('--group', default=None,
                     help='experiment name; runs of a group differ only in --seed')
    obs = ap.add_argument_group('observability (all opt-in)')
    obs.add_argument('--ckpt-every', type=int, default=0,
                     help='updates between resumable checkpoints in ckpt/ (0 = off)')
    obs.add_argument('--ckpt-keep', type=int, default=0,
                     help='keep only the last K checkpoints (0 = keep all) ...')
    obs.add_argument('--ckpt-thin', type=int, default=0,
                     help='... plus one every T updates (0 = none)')
    obs.add_argument('--replay-every', type=int, default=0,
                     help='updates between replay recordings of the current policy (0 = off)')
    obs.add_argument('--replay-n', type=int, default=8,
                     help='replays per recording (half greedy, half sampled)')
    obs.add_argument('--resume', nargs='?', const='latest', default=None,
                     help='resume the run from CKPT (default: its latest checkpoint)')
    return ap.parse_args(argv)


# Flags that may differ on --resume (operational, not part of the experiment).
RESUME_OVERRIDES = ('total_steps', 'device', 'threads', 'log_every', 'eval_every',
                    'ckpt_every', 'ckpt_keep', 'ckpt_thin', 'replay_every', 'replay_n',
                    'runs_dir', 'eval_episodes', 'eval_envs')


def _explicit_flags(argv) -> set:
    """Destinations given explicitly on the command line (``--flag`` or
    ``--flag=value``), used to tell overrides from defaults on ``--resume``."""
    argv = sys.argv[1:] if argv is None else list(argv)
    return {tok[2:].split('=', 1)[0].replace('-', '_')
            for tok in argv if tok.startswith('--')}


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class StopRequested:
    """SIGTERM/SIGINT set a flag; the loop stops after the current update. A
    second signal falls back to the default behaviour (immediate exit)."""

    def __init__(self):
        self.signal = None
        self._previous = {}

    def install(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._previous[sig] = signal.signal(sig, self._handler)
            except ValueError:              # not in the main thread
                pass

    def _handler(self, signum, frame):
        if self.signal is not None:
            signal.signal(signum, self._previous.get(signum, signal.SIG_DFL))
            raise KeyboardInterrupt
        self.signal = signum
        print(f'signal {signum}: stopping after the current update', flush=True)

    def restore(self):
        for sig, handler in self._previous.items():
            signal.signal(sig, handler)

    def __bool__(self):
        return self.signal is not None


def _rng_state():
    state = {'torch': torch.get_rng_state(), 'numpy': np.random.get_state()}
    if torch.backends.mps.is_available():
        try:
            state['mps'] = torch.mps.get_rng_state()
        except (AttributeError, RuntimeError):
            pass
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    if 'mps' in state and torch.backends.mps.is_available():
        try:
            torch.mps.set_rng_state(state['mps'])
        except (AttributeError, RuntimeError):
            pass
    if 'cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _prune_checkpoints(ckpt_dir: Path, keep: int, thin: int):
    if keep <= 0:
        return
    found = sorted(ckpt_dir.glob('update_*.pt'))
    for path in found[:-keep]:
        u = int(path.stem.split('_')[1])
        if thin and u % thin == 0:
            continue
        path.unlink(missing_ok=True)


def latest_checkpoint(run_dir: Path):
    found = sorted((run_dir / 'ckpt').glob('update_*.pt'))
    return found[-1] if found else None


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _run_name(args):
    return args.name or (f'{args.preset or "levels" + args.levels}'
                         f'_n{args.num_levels if args.preset else ""}_s{args.seed}')


def _replay_refs(args, update, n):
    """Deterministic levels for ``--replay-every``: training levels, plus
    held-out seeds (every other replay) in pcg mode."""
    from .rollout import builtin_ref, preset_refs
    refs = []
    for k in range(n):
        mix = (update * 7919 + k * 104729)
        if args.preset is None:
            nums = _parse_levels(args.levels)
            refs.append(builtin_ref(nums[(update + k) % len(nums)]))
        elif k % 2 == 1:
            refs += preset_refs(args.preset, TEST_START_SEED + mix % max(1, args.eval_levels), 1)
        elif args.num_levels:
            refs += preset_refs(args.preset, args.start_seed + mix % args.num_levels, 1)
        else:
            refs += preset_refs(args.preset, mix % TEST_START_SEED, 1)
    return refs


def _record_policy_replays(agent, cpu_agent, args, update, env_steps, out, runs_root,
                           n, ckpt_rel):
    """Numbered replays of the current policy (CPU copy; private RNG)."""
    from .rollout import Policy, record_replays
    cpu_agent.load_state_dict({k: v.detach().cpu() for k, v in agent.state_dict().items()})
    cpu_agent.eval()
    policy = Policy(cpu_agent, vars(args), str(out / 'model.pt'),
                    {'update': update, 'env_steps': env_steps})
    source = {'kind': 'checkpoint', 'update': update, 'env_steps': env_steps,
              'ckpt': ckpt_rel}
    refs = _replay_refs(args, update, n)
    greedy = [r for k, r in enumerate(refs) if (k // (2 if args.preset else 1)) % 2 == 0]
    sampled = [r for k, r in enumerate(refs) if (k // (2 if args.preset else 1)) % 2 == 1]
    recs = []
    for flag, batch in ((True, greedy), (False, sampled)):
        if batch:
            recs += record_replays(policy, batch, out, runs_root, greedy=flag,
                                   seed=args.seed * 1_000_003 + update, n_envs=len(batch),
                                   source=source, tags=['train-replay'])
    return recs


def train(args):
    explicit = getattr(args, '_explicit', set())
    runs_root = Path(args.runs_dir) if args.runs_dir else ROOT / 'runs'
    resume_ckpt = None
    if args.resume is not None:
        name = _run_name(args)
        run_dir = runs_root / name
        if args.resume == 'latest':
            path = latest_checkpoint(run_dir)
            if path is None:
                raise SystemExit(f'--resume: no checkpoint in {run_dir / "ckpt"}')
        else:
            path = Path(args.resume)
            path = path if path.is_absolute() else run_dir / path
        resume_ckpt = torch.load(path, map_location='cpu', weights_only=False)
        # The experiment is the checkpoint's; only operational flags given
        # explicitly now override it.
        keep = {k: getattr(args, k) for k in ('resume', 'name', 'runs_dir')}
        for key, value in resume_ckpt['config'].items():
            if key.startswith('_'):
                continue
            if key in RESUME_OVERRIDES and key in explicit:
                continue
            setattr(args, key, value)
        for key, value in keep.items():
            setattr(args, key, value)
        args.name = name
        resumed_from = str(path.relative_to(run_dir)) if path.is_relative_to(run_dir) else str(path)
    else:
        resumed_from = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device != 'auto' else
                          'mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)
    name = _run_name(args)
    out = runs_root / name
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    (out / 'config.json').write_text(json.dumps(config, indent=2))

    envs = make_vec_env(level_source(args), args.num_envs, args)
    grid_shape = (len(GRID_CHANNELS), 21, 31)
    agent = Agent(grid_shape, len(VEC_FIELDS), 6).to(device)
    optim = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    T, N = args.num_steps, args.num_envs
    batch = T * N
    mb = batch // args.minibatches
    updates = max(1, args.total_steps // batch)
    buf_grid = torch.zeros((T, N, *grid_shape), dtype=torch.uint8)
    buf_vec = torch.zeros((T, N, len(VEC_FIELDS)))
    buf_act = torch.zeros((T, N), dtype=torch.long)
    buf_logp = torch.zeros((T, N))
    buf_rew = torch.zeros((T, N))
    buf_done = torch.zeros((T, N))
    buf_val = torch.zeros((T, N))

    start_update, episodes, seconds_before = 0, 0, 0.0
    recent = deque(maxlen=200)            # (return, won, has_gem)
    recent_len = deque(maxlen=200)        # engine steps per finished episode
    if resume_ckpt is not None:
        agent.load_state_dict(resume_ckpt['model'])
        optim.load_state_dict(resume_ckpt['optimizer'])
        start_update = int(resume_ckpt['update'])
        episodes = int(resume_ckpt.get('episodes', 0))
        seconds_before = float(resume_ckpt.get('train_seconds', 0.0))
        recent.extend(tuple(x) for x in resume_ckpt.get('recent', ()))
        recent_len.extend(resume_ckpt.get('recent_len', ()))

    # Rollout buffers live on the CPU; the network runs on ``device``.
    # On resume the envs restart (episodes in flight are not checkpointed),
    # with a seed that depends on the update so levels do not repeat.
    obs, _ = envs.reset(seed=args.seed if resume_ckpt is None else args.seed + start_update)
    if resume_ckpt is not None and 'rng' in resume_ckpt:
        _set_rng_state(resume_ckpt['rng'])
    grid, vec = to_tensors(obs, 'cpu')
    ep_ret = np.zeros(N)
    header = ['update', 'env_steps', 'sps', 'episodes', 'return', 'success',
              'gem_rate', 'pg_loss', 'v_loss', 'entropy', 'approx_kl',
              'eval_train', 'eval_test']
    if resume_ckpt is not None and (out / 'progress.csv').exists():
        log = open(out / 'progress.csv', 'a', newline='')
        writer = csv.writer(log)
    else:
        log = open(out / 'progress.csv', 'w', newline='')
        writer = csv.writer(log)
        writer.writerow(header)
        if (out / 'metrics.jsonl').exists():
            (out / 'metrics.jsonl').unlink()
    start = time.perf_counter()
    started_at = now_iso()
    status = {'schema': 1, 'name': name, 'group': args.group, 'state': 'running',
              'pid': os.getpid(), 'started_at': started_at, 'updated_at': started_at,
              'update': start_update, 'updates_total': updates,
              'env_steps': start_update * batch, 'total_steps': args.total_steps,
              'sps': None, 'device': str(device), 'git_commit': _git_commit(),
              'error': None, 'resumed_from': resumed_from, 'log_interval_s': None,
              'last_ckpt': None, 'ckpt_every': args.ckpt_every,
              'replay_every': args.replay_every}
    last_status = time.time()

    def write_status(**changes):
        nonlocal last_status
        t = time.time()
        if changes.get('state', 'running') == 'running':
            status['log_interval_s'] = round(t - last_status, 3)
        last_status = t
        status.update(changes, updated_at=now_iso())
        atomic_write_json(out / 'status.json', status)

    write_status()
    stop = StopRequested()
    stop.install()
    ckpt_dir = out / 'ckpt'
    cpu_agent = None
    replay_n, replay_seconds = args.replay_n, 0.0
    update = start_update
    last_logged = None

    def save_checkpoint(u):
        path = ckpt_dir / f'update_{u:06d}.pt'
        _atomic_torch_save({
            'model': {k: v.cpu() for k, v in agent.state_dict().items()},
            'optimizer': _to_cpu(optim.state_dict()),
            'update': u, 'env_steps': u * batch, 'episodes': episodes,
            'updates_total': updates, 'recent': list(recent), 'recent_len': list(recent_len),
            'config': config, 'rng': _rng_state(),
            'train_seconds': seconds_before + time.perf_counter() - start}, path)
        _prune_checkpoints(ckpt_dir, args.ckpt_keep, args.ckpt_thin)
        status['last_ckpt'] = str(path.relative_to(out))
        return str(path.relative_to(out))

    try:
        for update in range(start_update + 1, updates + 1):
            frac = 1.0 - (update - 1) / updates
            optim.param_groups[0]['lr'] = frac * args.lr
            for t in range(T):
                buf_grid[t], buf_vec[t] = grid, vec
                with torch.no_grad():
                    action, logp, _, value = agent.act(grid.to(device), vec.to(device))
                action = action.cpu()
                buf_act[t], buf_logp[t], buf_val[t] = action, logp.cpu(), value.cpu()
                obs, reward, term, trunc, info = envs.step(action.numpy())
                ep_ret += reward
                reward = reward.astype(np.float32)
                done = term | trunc
                for i in np.flatnonzero(done):
                    fi = info['final_info']
                    recent.append((ep_ret[i], bool(fi['is_success'][i]), bool(fi['has_gem'][i])))
                    recent_len.append(int(fi['step_count'][i]))
                    ep_ret[i] = 0.0
                    episodes += 1
                    if trunc[i] and not term[i]:      # time limit: bootstrap
                        fo = final_obs_at(info['final_obs'], i)
                        with torch.no_grad():
                            fg, fv = to_tensors({k: fo[k][None] for k in ('grid', 'vec')}, device)
                            reward[i] += args.gamma * agent.value(fg, fv).item()
                buf_rew[t] = torch.as_tensor(reward)
                buf_done[t] = torch.as_tensor(done, dtype=torch.float32)
                grid, vec = to_tensors(obs, 'cpu')

            with torch.no_grad():
                next_value = agent.value(grid.to(device), vec.to(device)).cpu()
            adv = compute_gae(buf_rew, buf_done, buf_val, next_value,
                              args.gamma, args.gae_lambda)
            ret = adv + buf_val

            b_grid = buf_grid.reshape(batch, *grid_shape).to(device)
            b_vec = buf_vec.reshape(batch, -1).to(device)
            b_act, b_logp = buf_act.reshape(-1).to(device), buf_logp.reshape(-1).to(device)
            b_adv, b_ret = adv.reshape(-1).to(device), ret.reshape(-1).to(device)
            for _ in range(args.epochs):
                perm = torch.randperm(batch, device=device)
                for s in range(0, batch, mb):
                    idx = perm[s:s + mb]
                    _, logp, ent, val = agent.act(b_grid[idx], b_vec[idx], b_act[idx])
                    ratio = (logp - b_logp[idx]).exp()
                    a = b_adv[idx]
                    a = (a - a.mean()) / (a.std() + 1e-8)
                    pg = torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                    v_loss = 0.5 * ((val - b_ret[idx]) ** 2).mean()
                    entropy = ent.mean()
                    loss = pg + args.vf_coef * v_loss - args.ent_coef * entropy
                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optim.step()
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - (logp - b_logp[idx])).mean().item()

            eval_train = eval_test = ''
            if args.eval_every and (update % args.eval_every == 0):
                if args.preset is not None:
                    eval_train = evaluate(agent, level_source(args), args.eval_episodes,
                                          args, device)['success_rate']
                    eval_test = evaluate(agent, level_source(args, test=True),
                                         args.eval_episodes, args, device)['success_rate']
                else:
                    eval_train = evaluate(agent, _parse_levels(args.levels),
                                          args.eval_episodes, args, device)['success_rate']
            steps = update * batch
            if update % args.log_every == 0 or update == updates or eval_train != '':
                sps = (update - start_update) * batch / (time.perf_counter() - start)
                r = np.array(recent, dtype=float) if recent else np.zeros((1, 3))
                row = [update, steps, round(sps), episodes, r[:, 0].mean(), r[:, 1].mean(),
                       r[:, 2].mean(), pg.item(), v_loss.item(), entropy.item(), approx_kl,
                       eval_train, eval_test]
                writer.writerow(row)
                log.flush()
                last_logged = update
                print(f'upd {update}/{updates} steps {steps} sps {sps:.0f} '
                      f'ret {r[:, 0].mean():.3f} win {r[:, 1].mean():.2f} gem {r[:, 2].mean():.2f} '
                      f'ent {entropy.item():.2f} kl {approx_kl:.4f}'
                      + (f' eval {eval_train}/{eval_test}' if eval_train != '' else ''),
                      flush=True)
                rec = dict(zip(header, row))
                rec.update({k: (None if v == '' else v) for k, v in
                            (('eval_train', eval_train), ('eval_test', eval_test))})
                rec.update(
                    t=round(seconds_before + time.perf_counter() - start, 3),
                    lr=optim.param_groups[0]['lr'],
                    ep_len_mean=float(np.mean(recent_len)) if recent_len else None,
                    action_hist=torch.bincount(buf_act.reshape(-1), minlength=6).tolist(),
                    time_left_frac=float(buf_vec[:, :, 11].mean()))
                append_jsonl(out / 'metrics.jsonl', [rec])
                write_status(update=update, env_steps=steps, sps=round(sps))

            ckpt_rel = None
            if args.ckpt_every and update % args.ckpt_every == 0:
                ckpt_rel = save_checkpoint(update)
            if args.replay_every and update % args.replay_every == 0 and replay_n > 0:
                if cpu_agent is None:
                    cpu_agent = Agent(grid_shape, len(VEC_FIELDS), 6)
                t0 = time.perf_counter()
                _record_policy_replays(agent, cpu_agent, args, update, steps, out, runs_root,
                                       replay_n, ckpt_rel)
                replay_seconds += time.perf_counter() - t0
                elapsed = time.perf_counter() - start
                if replay_seconds > 0.03 * elapsed and replay_n > 2:
                    replay_n = max(2, replay_n // 2)
                    print(f'replays took {100 * replay_seconds / elapsed:.1f}% of training '
                          f'time; --replay-n reduced to {replay_n}', flush=True)
            if stop:
                break
    except BaseException as exc:
        log.close()
        envs.close()
        stop.restore()
        last = traceback.format_exception_only(type(exc), exc)[-1].strip()
        write_status(state='stopped' if isinstance(exc, KeyboardInterrupt) else 'failed',
                     error=last)
        raise

    stop.restore()
    log.close()
    envs.close()
    train_seconds = seconds_before + time.perf_counter() - start
    torch.save({'model': {k: v.cpu() for k, v in agent.state_dict().items()},
                'config': config}, out / 'model.pt')
    if stop:
        if status.get('last_ckpt') != f'ckpt/update_{update:06d}.pt':
            save_checkpoint(update)
        result = {'partial': True, 'stopped_at_update': update, 'updates_total': updates,
                  'env_steps': update * batch, 'train_seconds': train_seconds,
                  'last_logged_update': last_logged}
        (out / 'eval.json').write_text(json.dumps(result, indent=2))
        write_status(state='stopped', update=update, env_steps=update * batch)
        print(json.dumps(result, indent=2))
        return result
    result = evaluate_all(agent, args, device)
    result['train_seconds'] = train_seconds
    (out / 'eval.json').write_text(json.dumps(result, indent=2))
    write_status(state='finished')
    print(json.dumps(result, indent=2))
    return result


def _mark_failed(args, exc):
    """Record a failure that happened before the training loop owned
    ``status.json`` (bad arguments, environment construction, ...)."""
    runs_root = Path(args.runs_dir) if args.runs_dir else ROOT / 'runs'
    out = runs_root / _run_name(args)
    if not out.is_dir():
        return
    path = out / 'status.json'
    try:
        status = json.loads(path.read_text())
    except (OSError, ValueError):
        status = {'schema': 1, 'name': out.name, 'group': args.group, 'pid': os.getpid(),
                  'started_at': now_iso()}
    if status.get('state') in ('failed', 'stopped', 'finished') and status.get('pid') == os.getpid():
        return
    status.update(state='failed', updated_at=now_iso(), pid=os.getpid(),
                  error=traceback.format_exception_only(type(exc), exc)[-1].strip())
    atomic_write_json(path, status)


def main(argv=None):
    args = parse_args(argv)
    args._explicit = _explicit_flags(argv)
    try:
        train(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        _mark_failed(args, exc)
        raise


if __name__ == '__main__':
    main()
