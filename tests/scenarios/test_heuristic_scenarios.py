"""Cases that pin how the heuristic reasons, against the frozen fixture.

Two things are worth pinning exactly rather than measuring in aggregate. The
valuation must rank hands by *coins*, not by cards-from-completion — the payouts
are far enough apart that the two orderings genuinely disagree. And the defensive
term must produce real numbers, because "expected coins billed to me" is only
comparable to "expected coins I win" if it is actually denominated in coins.

Fixture payouts (frozen): triple 100 mixed / 700 mono; two-group 50 / 150;
four-group 400 / 800; bonus +50 per scoring copy, and the bonus holomem is Charlie.
"""

from __future__ import annotations

import pytest

from pokajan.agents.heuristic import HeuristicAgent, completion_probability
from pokajan.core.actions import DecisionType, legal_mask
from pokajan.core.cards import Counts
from pokajan.envs.belief import Particle
from pokajan.server.protocol import DecisionRequest, PublicState, SeatView

pytestmark = pytest.mark.scenario


def state_with(rules, hand: Counts, *, deck_remaining: int | None = None) -> PublicState:
    n = rules.cards.n_slots
    players = rules.play.players
    return PublicState(
        game_id="scenario",
        turn_index=4,
        viewer=0,
        hand=list(hand),
        seats=[
            SeatView(
                seat=s,
                coins=rules.play.initial_coins,
                hand_size=sum(hand) if s == 0 else rules.play.hand_limit,
                discards=[0] * n,
                calls_made=0,
                coins_won=0,
                coins_paid=0,
            )
            for s in range(players)
        ],
        deck_remaining=deck_remaining if deck_remaining is not None else 20,
        table=[0] * n,
        scored=[0] * n,
        last_discard_slot=None,
        last_discard_seat=None,
        current_seat=0,
        phase="AWAIT_DISCARD",
        bonus_character=rules.bonus_character,
        recent_discards=[[] for _ in range(players)],
    )


def discard_request(rules, hand: Counts) -> DecisionRequest:
    return DecisionRequest(
        game_id="scenario",
        seat=0,
        decision="DISCARD",
        legal_mask=legal_mask(
            rules, DecisionType.DISCARD, hand, bonus_character=rules.bonus_character
        ),
        state=state_with(rules, hand),
    )


# --------------------------------------------------------------- valuation --


def test_claims_cannot_advance_a_hand_that_is_two_cards_short(fixture_rules):
    """A claim is legal only when the claimed card *completes* the hand.

    So no amount of opponent discards moves a hand from two short to one short.
    Modelling claims as generic extra draws would quietly inflate every long-shot
    hand, and long shots are exactly where the big payouts live — a monochrome
    five-group pays more than the starting stack.
    """
    unseen = [1.0] * 40
    total = 40.0

    one_short_draws_only = completion_probability(
        [(0,)], unseen, total, my_draws=10, claim_ops=0.0
    )
    two_short_with_endless_claims = completion_probability(
        [(0,), (1,)], unseen, total, my_draws=10, claim_ops=10_000.0
    )

    assert two_short_with_endless_claims == pytest.approx(one_short_draws_only, rel=1e-6)


def test_claims_help_a_hand_that_is_one_card_short(fixture_rules):
    unseen = [1.0] * 40
    without = completion_probability([(0,)], unseen, 40.0, my_draws=10, claim_ops=0.0)
    with_claims = completion_probability([(0,)], unseen, 40.0, my_draws=10, claim_ops=30.0)

    assert with_claims > without


def test_probability_tracks_how_many_copies_are_left(fixture_rules):
    scarce = [0.0] * 40
    scarce[0] = 1.0
    plentiful = [0.0] * 40
    plentiful[0] = 3.0
    gone = [0.0] * 40

    low = completion_probability([(0,)], scarce, 40.0, my_draws=8, claim_ops=5.0)
    high = completion_probability([(0,)], plentiful, 40.0, my_draws=8, claim_ops=5.0)

    assert 0.0 < low < high < 1.0
    # A card with no copies left is unreachable, not merely unlikely.
    assert completion_probability([(0,)], gone, 40.0, my_draws=99, claim_ops=99.0) == 0.0


def test_does_not_chase_the_glamorous_hand_over_the_likely_one(fixture_rules):
    """Coins *times probability*, not coins.

    The hand below is one card short of three things at once. Two are pairs headed
    for monochrome triples worth 700 and 850 — but each needs one *exact* remaining
    card, of which barely any are left. The third is the four-group Right, worth
    450 and needing any colour of a single character, which is close to a
    formality.

    An agent ranking by payout keeps the 850. An agent ranking by cards-from-
    completion cannot tell them apart at all: all three are one away. Only expected
    coins gets this right, and it gets it right by keeping the *cheap* hand.
    """
    space = fixture_rules.cards
    blue, orange, pink = 0, 1, 2

    hand = space.zeros()
    hand[space.slot("c", blue)] = 2      # monochrome triple Charlie: 850, needs blue c
    hand[space.slot("a", orange)] = 2    # monochrome triple Alpha:   700, needs orange a
    hand[space.slot("d", blue)] = 1      # with f, one short of group Right: 450
    hand[space.slot("f", pink)] = 1

    agent = HeuristicAgent(fixture_rules, seed=5, particles=0, defend=False)
    action = agent.act(discard_request(fixture_rules, hand)).action

    assert action in (space.slot("c", blue), space.slot("a", orange)), (
        f"broke up the nearly-certain group to chase a long shot: "
        f"{space.describe_slot(action)}"
    )


def test_ranking_follows_what_is_actually_left(fixture_rules):
    """The same hand is worth different amounts depending on the deck.

    This is the belief earning its keep at the valuation layer: two hands one card
    from completion, priced by how many of that card the deck is believed to still
    hold. Supplying the availability directly keeps the test exact rather than at
    the mercy of a sampled posterior.
    """
    space = fixture_rules.cards
    blue = 0
    n = space.n_slots

    agent = HeuristicAgent(fixture_rules, seed=1, particles=0, defend=False)
    agent.start_game(state_with(fixture_rules, space.zeros()))

    hand = space.zeros()
    hand[space.slot("c", blue)] = 2      # one blue Charlie from 850

    plentiful = [0.01] * n
    plentiful[space.slot("c", blue)] = 3.0
    scarce = [0.01] * n
    scarce[space.slot("c", blue)] = 0.01

    rich = agent.hand_value(hand, (plentiful, 40.0, 8.0, 5.0))
    poor = agent.hand_value(hand, (scarce, 40.0, 8.0, 5.0))

    assert rich > poor
    # The ceiling is the payout itself: probabilities may approach 1 but a hand is
    # never worth more than what it pays.
    assert rich <= 850


# ----------------------------------------------------------------- defence --


def test_danger_is_measured_in_coins(fixture_rules):
    """The defensive term must be the actual bill, hand by hand.

    Discard safety carries unusual weight in this game because a claim bills the
    discarder *alone* — there is no pot to share the damage. So `danger` returns
    the payout the claimer would collect, and the discard rule can subtract it
    from the offensive value directly with no fudge factor between them.

    Here one opponent holds two blue Alphas and nothing else. Three candidate
    discards, three different bills.
    """
    space = fixture_rules.cards
    blue = 0
    n = space.n_slots

    opponent = space.zeros()
    opponent[space.slot("a", blue)] = 2

    hand = space.zeros()
    hand[space.slot("a", blue)] = 1
    hand[space.slot("b", blue)] = 1
    hand[space.slot("f", blue)] = 1

    state = state_with(fixture_rules, hand)
    agent = HeuristicAgent(fixture_rules, seed=1, particles=0)
    agent.start_game(state)

    particle = Particle(
        hands=[hand, opponent, space.zeros(), space.zeros()],
        deck=space.zeros(),
        composition=[0] * n,
        weight=1.0,
    )
    candidates = [space.slot("a", blue), space.slot("b", blue), space.slot("f", blue)]
    danger = agent.danger(candidates, [particle], state)

    # A third blue Alpha completes a monochrome triple: 700, and Alpha is not the
    # bonus holomem so nothing is added.
    assert danger[space.slot("a", blue)] == 700
    # A blue Bravo completes the monochrome two-group Left with their Alpha: 150.
    assert danger[space.slot("b", blue)] == 150
    # Foxtrot needs Charlie, Delta and Echo alongside it, and they hold none.
    assert danger[space.slot("f", blue)] == 0


def test_danger_is_an_expectation_over_worlds(fixture_rules):
    """Weighted by belief, so an unlikely disaster is priced, not feared."""
    space = fixture_rules.cards
    blue = 0
    n = space.n_slots

    dangerous = space.zeros()
    dangerous[space.slot("a", blue)] = 2
    harmless = space.zeros()

    hand = space.zeros()
    hand[space.slot("a", blue)] = 1

    state = state_with(fixture_rules, hand)
    agent = HeuristicAgent(fixture_rules, seed=1, particles=0)
    agent.start_game(state)

    particles = [
        Particle([hand, dangerous, space.zeros(), space.zeros()], space.zeros(), [0] * n, 0.25),
        Particle([hand, harmless, space.zeros(), space.zeros()], space.zeros(), [0] * n, 0.75),
    ]
    danger = agent.danger([space.slot("a", blue)], particles, state)

    assert danger[space.slot("a", blue)] == pytest.approx(0.25 * 700)


def test_only_one_opponent_can_win_a_claim(fixture_rules):
    """Danger is a maximum across opponents, never a sum.

    Two players may both be able to claim, but only the higher payout collects, so
    adding their bills together would overstate the risk of every discard and make
    the agent hoard cards it should be throwing.
    """
    space = fixture_rules.cards
    blue = 0
    n = space.n_slots

    triple_threat = space.zeros()
    triple_threat[space.slot("a", blue)] = 2       # would claim 700
    group_threat = space.zeros()
    group_threat[space.slot("b", blue)] = 1        # would claim 150

    hand = space.zeros()
    hand[space.slot("a", blue)] = 1

    state = state_with(fixture_rules, hand)
    agent = HeuristicAgent(fixture_rules, seed=1, particles=0)
    agent.start_game(state)

    particle = Particle(
        [hand, triple_threat, group_threat, space.zeros()], space.zeros(), [0] * n, 1.0
    )
    danger = agent.danger([space.slot("a", blue)], [particle], state)

    assert danger[space.slot("a", blue)] == 700
