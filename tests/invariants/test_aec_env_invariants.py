"""Properties of the AEC wrapper.

The one that matters is reward telescoping: the dense per-event coin rewards must
sum to exactly the terminal outcome. If they do not, the shaping is biased and the
agent is being trained toward something other than winning.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pokajan.agents.base import RandomAgent
from pokajan.core.rules import Rules
from pokajan.envs.aec_env import PokajanAECEnv
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def run(env, seed: int) -> dict[int, float]:
    """Play out an env with random legal actions, returning summed rewards."""
    agents = [RandomAgent(seed=seed + i) for i in env.possible_agents]
    totals = {a: 0.0 for a in env.possible_agents}
    guard = 0

    while env.agent_selection is not None:
        request, _, _, _, _ = env.last()
        action = agents[request.seat].act(request).action
        env.step(action)
        for seat in env.possible_agents:
            totals[seat] += env.rewards[seat]
        guard += 1
        assert guard < 20_000, "env is not progressing"
    return totals


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_dense_rewards_sum_to_the_final_outcome(cfg, seed):
    """Potential-based shaping: the parts must add up to the whole.

    Each per-event reward is a coin delta over the starting stack, so summing them
    across the game has to reproduce (final - initial) / initial exactly. Any drift
    means rewards are being invented or dropped somewhere in the payment path.
    """
    rules = Rules.from_dict(cfg)
    env = PokajanAECEnv(rules, seed=seed)
    totals = run(env, seed)

    initial = rules.play.initial_coins
    final = env.engine.final_coins()
    for seat in env.possible_agents:
        expected = (final[seat] - initial) / initial
        assert totals[seat] == pytest.approx(expected, abs=1e-9)


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_env_terminates_and_reports_a_result(cfg, seed):
    rules = Rules.from_dict(cfg)
    env = PokajanAECEnv(rules, seed=seed)
    run(env, seed)

    assert env.agent_selection is None
    assert all(env.terminations.values())
    info = env.infos[0]
    assert info["end_reason"] in {"deck_empty", "bankrupt"}
    assert len(info["final_coins"]) == rules.play.players
    assert sorted(info["standings"]) == list(range(rules.play.players))

    _, _, terminated, _, _ = env.last()
    assert terminated
    with pytest.raises(ValueError, match="game is over"):
        env.step(0)


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_only_masked_actions_are_offered(cfg, seed):
    rules = Rules.from_dict(cfg)
    env = PokajanAECEnv(rules, seed=seed)
    n = rules.cards.n_slots + 2
    guard = 0

    while env.agent_selection is not None:
        mask = env.legal_mask()
        assert mask is not None and len(mask) == n
        assert any(mask)
        request, _, _, _, _ = env.last()
        assert request.seat == env.agent_selection
        env.step(mask.index(True))
        guard += 1
        assert guard < 20_000


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_clone_diverges_without_touching_the_original(cfg, seed):
    rules = Rules.from_dict(cfg)
    env = PokajanAECEnv(rules, seed=seed)
    agents = [RandomAgent(seed=seed + i) for i in env.possible_agents]

    for _ in range(10):
        if env.agent_selection is None:
            return
        request, _, _, _, _ = env.last()
        env.step(agents[request.seat].act(request).action)

    if env.agent_selection is None:
        return

    fork = env.clone()
    assert fork.agent_selection == env.agent_selection
    assert fork.engine.state.coins == env.engine.state.coins

    before = env.engine.state.coins[:]
    run(fork, seed + 500)
    assert env.engine.state.coins == before
    assert env.agent_selection is not None
