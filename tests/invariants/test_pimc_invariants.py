"""Properties of the determinized search.

PIMC does not currently beat the heuristic it wraps — that is measured, and the
reasoning is in the README. These tests are about the machinery being *correct*
regardless, because the determinizer and the fast policy underneath it are kept
for M5 and for any later search, and a broken search that merely looks weak is
indistinguishable from a working search on a hard game.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from pokajan.agents.heuristic import HeuristicAgent
from pokajan.agents.pimc import PIMCAgent
from pokajan.core.actions import call_action, pass_action
from pokajan.core.engine import Engine
from pokajan.core.rules import Rules
from pokajan.envs.driver import play_game
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

# Search is expensive; these settings keep a whole game affordable per example.
TINY = dict(determinizations=2, candidates=2, rollout_depth=20)

FEW = settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@given(rules_configs())
@FEW
def test_search_plays_a_legal_game(config):
    rules = Rules.from_dict(config)
    engine = Engine.new_game(rules, seed=51)
    agents = [PIMCAgent(rules, seed=0, **TINY)] + [
        HeuristicAgent(rules, seed=i, particles=4) for i in range(1, rules.play.players)
    ]

    result = play_game(engine, agents)

    assert engine.finished
    assert engine.state.cards_in_play() == rules.deck_size
    assert sum(result.final_coins) == (
        rules.play.initial_coins * rules.play.players + result.coins_minted
    )


@given(rules_configs())
@FEW
def test_search_never_proposes_an_illegal_action(config):
    rules = Rules.from_dict(config)
    engine = Engine.new_game(rules, seed=52)
    agents = [PIMCAgent(rules, seed=0, **TINY)] + [
        HeuristicAgent(rules, seed=i, particles=4) for i in range(1, rules.play.players)
    ]

    checked = 0
    while not engine.finished and checked < 40:
        pending = engine.pending_decisions()
        if not pending:
            break
        responses = []
        for request in pending:
            response = agents[request.seat].act(request)
            assert request.legal_mask[response.action]
            responses.append(response)
            checked += 1
        engine.submit(responses)

    assert checked > 0


def test_a_forced_move_skips_the_search(fixture_rules):
    """With one legal action there is nothing to decide, and search is not free.

    Worth pinning: a decision that runs dozens of rollouts to rediscover the only
    move it was allowed to make would not fail any other test here, it would just
    make every matchup against PIMC several times slower.
    """
    engine = Engine.new_game(fixture_rules, seed=61)
    agent = PIMCAgent(fixture_rules, seed=0, **TINY)

    request = engine.pending_decisions()[0]
    forced = [False] * len(request.legal_mask)
    only = next(i for i, ok in enumerate(request.legal_mask) if ok)
    forced[only] = True
    forced_request = type(request)(
        game_id=request.game_id,
        seat=request.seat,
        decision=request.decision,
        legal_mask=forced,
        state=request.state,
        claimable_slot=request.claimable_slot,
    )

    calls = {"n": 0}
    original = agent.policy.choose

    def counted(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    agent.policy.choose = counted
    action = agent.act(forced_request).action

    assert action == only
    assert calls["n"] == 0, "search ran on a decision that had no alternatives"


def test_chain_context_follows_this_seats_own_move(fixture_rules):
    """Whether a chain owes a discard depends on how the chain began.

    That is not in PublicState, so the agent remembers its own last move instead of
    guessing. Getting it wrong changes the refill size inside every simulated
    chain, which would bias the search without ever raising an error.
    """
    engine = Engine.new_game(fixture_rules, seed=62)
    agent = PIMCAgent(fixture_rules, seed=0, **TINY)
    state = engine.public_state(0)
    call = call_action(fixture_rules)

    agent._respond(_request_of(engine, state, "CLAIM"), call, _decision("CLAIM"))
    assert agent._chain_from_claim is True

    agent._respond(_request_of(engine, state, "IN_TURN_CALL"), call, _decision("IN_TURN_CALL"))
    assert agent._chain_from_claim is False

    agent._respond(_request_of(engine, state, "CLAIM"), call, _decision("CLAIM"))
    assert agent._chain_from_claim is True
    # Passing on a claim starts no chain.
    agent._respond(
        _request_of(engine, state, "CLAIM"), pass_action(fixture_rules), _decision("CLAIM")
    )
    assert agent._chain_from_claim is False


def _decision(name: str):
    from pokajan.core.actions import DecisionType

    return DecisionType[name]


def _request_of(engine, state, decision: str):
    from pokajan.server.protocol import DecisionRequest

    return DecisionRequest(
        game_id=state.game_id,
        seat=0,
        decision=decision,
        legal_mask=[True] * (engine.state.rules.cards.n_slots + 2),
        state=state,
    )
