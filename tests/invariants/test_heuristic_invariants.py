"""Properties the heuristic agent must satisfy under any rules config.

An agent that is merely *bad* is fine — that is what the evaluation harness is
for. What these guard against is an agent that is *broken* in a way the engine
would absorb: proposing illegal moves, stalling a game, or reading state it should
not have. All three would show up as strange training results long before anyone
suspected the opponent policy.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from pokajan.agents.base import GreedyCallerAgent
from pokajan.agents.heuristic import HeuristicAgent, build_targets
from pokajan.core.engine import Engine
from pokajan.core.rules import Rules
from pokajan.envs.driver import play_game
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

# Enough particles to exercise the defensive path, few enough to run a whole game
# per Hypothesis example.
TEST_PARTICLES = 6

SLOW = settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@given(rules_configs())
@SLOW
def test_plays_a_legal_game_to_completion(config):
    rules = Rules.from_dict(config)
    engine = Engine.new_game(rules, seed=21)
    agents = [
        HeuristicAgent(rules, seed=i, particles=TEST_PARTICLES) for i in range(rules.play.players)
    ]

    result = play_game(engine, agents)

    assert engine.finished
    assert engine.state.cards_in_play() == rules.deck_size
    assert sum(result.final_coins) == (
        rules.play.initial_coins * rules.play.players + result.coins_minted
    )
    assert all(coins >= rules.end.coin_floor for coins in result.final_coins)


@given(rules_configs())
@SLOW
def test_never_proposes_an_illegal_action(config):
    """Checked before the engine sees it, so a fallback cannot hide a bug.

    `act` has a safety net that swaps an illegal choice for a legal one. That net
    exists so a bad heuristic degrades instead of crashing a training run — but it
    would also mask the bug forever, so the raw choice is what gets asserted here.
    """
    rules = Rules.from_dict(config)
    engine = Engine.new_game(rules, seed=22)
    agents = [
        HeuristicAgent(rules, seed=i, particles=TEST_PARTICLES) for i in range(rules.play.players)
    ]

    checked = 0
    while not engine.finished and checked < 60:
        pending = engine.pending_decisions()
        if not pending:
            break
        responses = []
        for request in pending:
            response = agents[request.seat].act(request)
            assert request.legal_mask[response.action], (
                f"{request.decision} produced illegal action {response.action}"
            )
            responses.append(response)
            checked += 1
        engine.submit(responses)

    assert checked > 0


@given(rules_configs())
@SLOW
def test_targets_cover_every_scoring_hand(config):
    """The valuation is a maximum over targets, so a missing one is invisible.

    It would not error, or even look wrong — the agent would simply never aim at
    that hand. Counting them against the roster is the only cheap way to notice.
    """
    rules = Rules.from_dict(config)
    space = rules.cards
    targets = build_targets(rules, rules.bonus_character)

    expected = (
        space.n_chars * (1 + space.n_colors)          # mixed + monochrome triples
        + space.n_groups * (1 + space.n_colors)       # mixed + monochrome groups
    )
    assert len(targets) == expected
    assert len({(t.kind, t.character, t.group, t.color) for t in targets}) == expected


@given(rules_configs())
@SLOW
def test_beliefs_are_not_shared_between_seats(config):
    """Four agents must not end up reading each other's cards.

    Sharing one belief across seats is an easy wiring mistake and produces an agent
    that is strong for entirely illegitimate reasons.
    """
    rules = Rules.from_dict(config)
    engine = Engine.new_game(rules, seed=23)
    agents = [HeuristicAgent(rules, seed=1, particles=TEST_PARTICLES) for _ in range(rules.play.players)]

    for _ in range(6):
        if engine.finished:
            break
        pending = engine.pending_decisions()
        if not pending:
            break
        engine.submit([agents[r.seat].act(r) for r in pending])

    acted = [(seat, a.belief) for seat, a in enumerate(agents) if a.belief is not None]
    assert acted, "no agent was asked to act"

    # Distinct objects, each bound to the seat that owns it, and each having seen
    # only that seat's cards.
    assert len({id(belief) for _, belief in acted}) == len(acted)
    state = engine.state
    for seat, belief in acted:
        assert belief.viewer == seat
        belief.observe(engine.public_state(seat))
        # Exactly this seat's own hand, the table, and the scored pile — nothing
        # from another hand and nothing from the deck.
        assert belief.seen == [
            state.hands[seat][slot] + state.table[slot] + state.scored[slot]
            for slot in range(rules.cards.n_slots)
        ]


def test_defence_changes_discards(fixture_rules):
    """The defensive term must actually move decisions.

    A weight that silently rounded to nothing would still pass every other test
    here and would quietly cost the agent the +160 coins/game the ablation in
    `train/evaluate.py` measures.
    """
    engine = Engine.new_game(fixture_rules, seed=31)
    defended = HeuristicAgent(fixture_rules, seed=2, particles=32)
    blind = HeuristicAgent(fixture_rules, seed=2, particles=32, defend=False)

    differences, decisions = 0, 0
    while not engine.finished and decisions < 40:
        pending = engine.pending_decisions()
        if not pending:
            break
        for request in pending:
            if request.decision == "DISCARD":
                decisions += 1
                if defended.act(request).action != blind.act(request).action:
                    differences += 1
        engine.submit(
            [GreedyCallerAgent(fixture_rules, seed=r.seat).act(r) for r in pending]
        )

    assert decisions > 0
    assert differences > 0, "defence never changed a discard — is DANGER_WEIGHT live?"
