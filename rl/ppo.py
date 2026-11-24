"""PPO baseline for GemEnv (single file, CleanRL style).

A small CNN reads the egocentric tile grid, an MLP reads the feature vector,
and both feed a shared actor-critic head. Truncated episodes (timeouts) are
bootstrapped from the value of their final observation.

    # built-in levels 1-6, gem pickup bonus
    python -m rl.ppo --levels 1-6 --gem-reward 0.5 --total-steps 2000000

    # generalisation: train on 200 generated levels, evaluate on unseen seeds
    python -m rl.ppo --preset tiny --num-levels 200 --total-steps 5000000

Writes ``runs/<name>/{config.json, progress.csv, eval.json, model.pt}``.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .env import GRID_CHANNELS, VEC_FIELDS, GemEnv

ROOT = Path(__file__).resolve().parents[1]
TEST_START_SEED = 1_000_000          # spec 11: [0, 1e6) train, [1e6, 2e6) test


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
    return ap.parse_args(argv)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device != 'auto' else
                          'mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}', flush=True)
    name = args.name or (f'{args.preset or "levels" + args.levels}'
                         f'_n{args.num_levels if args.preset else ""}_s{args.seed}')
    out = ROOT / 'runs' / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(args), indent=2))

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

    # Rollout buffers live on the CPU; the network runs on ``device``.
    obs, _ = envs.reset(seed=args.seed)
    grid, vec = to_tensors(obs, 'cpu')
    ep_ret = np.zeros(N)
    recent = deque(maxlen=200)            # (return, won, has_gem)
    log = open(out / 'progress.csv', 'w', newline='')
    writer = csv.writer(log)
    writer.writerow(['update', 'env_steps', 'sps', 'episodes', 'return', 'success',
                     'gem_rate', 'pg_loss', 'v_loss', 'entropy', 'approx_kl',
                     'eval_train', 'eval_test'])
    episodes, start = 0, time.perf_counter()

    for update in range(1, updates + 1):
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
        if update % args.log_every == 0 or update == updates or eval_train != '':
            steps = update * batch
            sps = steps / (time.perf_counter() - start)
            r = np.array(recent, dtype=float) if recent else np.zeros((1, 3))
            row = [update, steps, round(sps), episodes, r[:, 0].mean(), r[:, 1].mean(),
                   r[:, 2].mean(), pg.item(), v_loss.item(), entropy.item(), approx_kl,
                   eval_train, eval_test]
            writer.writerow(row)
            log.flush()
            print(f'upd {update}/{updates} steps {steps} sps {sps:.0f} '
                  f'ret {r[:, 0].mean():.3f} win {r[:, 1].mean():.2f} gem {r[:, 2].mean():.2f} '
                  f'ent {entropy.item():.2f} kl {approx_kl:.4f}'
                  + (f' eval {eval_train}/{eval_test}' if eval_train != '' else ''),
                  flush=True)

    log.close()
    envs.close()
    torch.save({'model': {k: v.cpu() for k, v in agent.state_dict().items()},
                'config': vars(args)}, out / 'model.pt')
    result = evaluate_all(agent, args, device)
    result['train_seconds'] = time.perf_counter() - start
    (out / 'eval.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main(argv=None):
    train(parse_args(argv))


if __name__ == '__main__':
    main()
