"""Properties of the fast policy, under any rules config.

The rollout policy is a second, independent implementation of two things the
engine already knows how to do: whether a discard can be claimed, and what the
claim would pay. It exists only because the engine's version allocates objects and
this one cannot afford to. That makes it exactly the kind of code that drifts —
a fast path that quietly disagrees with the slow one would show up as a search
recommending moves for reasons the rules do not support.

So the two are tested against each other directly, on random hands and random
rosters. If they ever disagree, the fast path is wrong.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pokajan.agents.rollout import (
    FastAgent,
    RolloutPolicy,
    _could_claim,
    claim_payout,
    play_out,
)
from pokajan.core.engine import Engine
from pokajan.core.evaluate import best_call, can_call_using
from pokajan.core.rules import Rules
from pokajan.envs.driver import play_game
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

SLOW = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

FEW = settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def build_hand(rules: Rules, draws: list[int]):
    """A legal hand from a list of arbitrary integers."""
    space = rules.cards
    hand = space.zeros()
    for value in draws:
        slot = value % space.n_slots
        if hand[slot] < space.max_per_color and sum(hand) < rules.play.hand_limit:
            hand[slot] += 1
    return hand


@given(
    rules_configs(),
    st.lists(st.integers(min_value=0, max_value=500), min_size=0, max_size=8),
    st.integers(min_value=0, max_value=500),
)
@SLOW
def test_fast_claim_payout_matches_the_engine(config, draws, pick):
    """The fast path must price a claim exactly as `best_call` does."""
    rules = Rules.from_dict(config)
    space = rules.cards
    bonus = rules.bonus_character

    hand = build_hand(rules, draws)
    slot = pick % space.n_slots
    if hand[slot] >= space.max_per_color:
        return

    plans = RolloutPolicy(rules)._character_targets(bonus)
    totals = space.char_totals(hand)

    probe = hand[:]
    probe[slot] += 1
    call = best_call(rules, probe, bonus_character=bonus, claimed=True, must_use=slot)
    expected = call.payout if call is not None else 0

    assert claim_payout(rules, plans, hand, totals, slot) == expected


@given(
    rules_configs(),
    st.lists(st.integers(min_value=0, max_value=500), min_size=0, max_size=8),
    st.integers(min_value=0, max_value=500),
)
@SLOW
def test_fast_eligibility_matches_the_engine(config, draws, pick):
    """...and must agree on whether a claim is legal at all.

    This is the rule that stops a made hand claiming any card it likes, so a fast
    path that got it wrong would hand the search a large illegitimate edge — the
    same bug that was caught in the engine at M1, reintroduced through the back
    door.
    """
    rules = Rules.from_dict(config)
    space = rules.cards
    bonus = rules.bonus_character

    hand = build_hand(rules, draws)
    slot = pick % space.n_slots
    if hand[slot] >= space.max_per_color:
        return

    plans = RolloutPolicy(rules)._character_targets(bonus)
    totals = space.char_totals(hand)

    probe = hand[:]
    probe[slot] += 1
    expected = can_call_using(rules, probe, slot, bonus_character=bonus)

    assert _could_claim(plans, totals, space.slot_char[slot]) is expected


@given(rules_configs())
@FEW
def test_rollouts_finish_and_conserve_cards(config):
    rules = Rules.from_dict(config)
    policy = RolloutPolicy(rules, seed=2)
    engine = Engine.new_game(rules, seed=41)

    coins = play_out(engine, policy)

    assert engine.finished
    assert engine.state.cards_in_play() == rules.deck_size
    assert sum(coins) == (
        rules.play.initial_coins * rules.play.players + engine.state.coins_minted
    )


@given(rules_configs())
@FEW
def test_fast_agent_plays_a_legal_game(config):
    rules = Rules.from_dict(config)
    engine = Engine.new_game(rules, seed=42)
    agents = [FastAgent(rules, seed=i, particles=4) for i in range(rules.play.players)]

    result = play_game(engine, agents)

    assert engine.finished
    assert engine.state.cards_in_play() == rules.deck_size
    assert sum(result.final_coins) == (
        rules.play.initial_coins * rules.play.players + result.coins_minted
    )


def test_rollout_policy_refuses_rules_it_would_misprice(fixture_rules):
    """A whole-hand bonus cannot be folded into a precomputed payout.

    Better to refuse than to search with quietly wrong numbers — the failure would
    look like a weak agent rather than a bug, which is the expensive kind.
    """
    config = dict(fixture_rules.raw)
    config["payouts"] = dict(config["payouts"])
    config["payouts"]["bonus"] = {"per_copy": 50, "applies_to": "whole_hand"}

    with pytest.raises(ValueError, match="scoring_set"):
        RolloutPolicy(Rules.from_dict(config))
