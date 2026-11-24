"""PPO baseline smoke test: a few tiny updates run end to end on the CPU and
write their outputs; learning itself is measured by the runs, not by tests."""
import json
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gymnasium")

from rl import ppo  # noqa: E402


@pytest.mark.parametrize('extra', [['--levels', '4'],
                                   ['--preset', 'tiny', '--num-levels', '3',
                                    '--eval-levels', '3']])
def test_ppo_runs_end_to_end(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(ppo, 'ROOT', tmp_path)
    args = ppo.parse_args([*extra, '--total-steps', '128', '--num-envs', '4',
                           '--num-steps', '16', '--minibatches', '2', '--epochs', '1',
                           '--eval-episodes', '2', '--eval-envs', '2', '--eval-every', '1',
                           '--device', 'cpu', '--threads', '1', '--name', 'smoke'])
    result = ppo.train(args)
    out = tmp_path / 'runs' / 'smoke'
    # status.json and metrics.jsonl are always written since spec 13 (§93.2);
    # the four original files keep their format.
    assert {p.name for p in out.iterdir()} == {'config.json', 'progress.csv',
                                               'eval.json', 'model.pt',
                                               'status.json', 'metrics.jsonl'}
    assert json.loads((out / 'status.json').read_text())['state'] == 'finished'
    assert json.loads((out / 'eval.json').read_text()) == result
    rows = (out / 'progress.csv').read_text().splitlines()
    assert len(rows) == 1 + 2                       # header + one row per update
    state = torch.load(out / 'model.pt')['model']
    agent = ppo.Agent((5, 21, 31), 12, 6)
    agent.load_state_dict(state)


def test_gae_does_not_bootstrap_across_episode_ends():
    # One env, three steps; the episode ends at t=1. With gamma=lambda=1 the
    # advantage at t=0 sums rewards 0..1 only, never the next episode's value.
    rew = torch.tensor([[1.0], [2.0], [4.0]])
    done = torch.tensor([[0.0], [1.0], [0.0]])
    adv = ppo.compute_gae(rew, done, torch.zeros(3, 1), torch.tensor([10.0]), 1.0, 1.0)
    assert adv.squeeze().tolist() == [3.0, 2.0, 14.0]


def test_gae_discounts_with_gamma_and_lambda():
    rew = torch.tensor([[0.0], [1.0]])
    val = torch.tensor([[0.5], [0.25]])
    adv = ppo.compute_gae(rew, torch.zeros(2, 1), val, torch.tensor([2.0]), 0.5, 0.5)
    d1 = 1.0 + 0.5 * 2.0 - 0.25          # 1.75
    d0 = 0.0 + 0.5 * 0.25 - 0.5          # -0.375
    assert adv.squeeze().tolist() == pytest.approx([d0 + 0.25 * d1, d1])
