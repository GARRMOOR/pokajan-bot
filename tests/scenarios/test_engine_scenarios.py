"""Hand-built positions exercising the rules that random play never reaches.

The smoke run makes the case for this file: across 1000 random games, not one
ended in bankruptcy and not a single coin was minted. The bankruptcy floor is a
confirmed rule sitting on an untravelled code path, so the only way it gets
tested is by rigging a position that forces it.

Fixture recap (tests/fixtures/rules_v1.yaml):
    characters a b c d e f    groups left={a,b}  right={c,d,e,f}   bonus c
    hand limit 5   triple 100   group 2->50, 4->400   mono x2   bonus x3
"""

from __future__ import annotations

import pytest

from pokajan.core.actions import call_action, pass_action
from pokajan.core.engine import Engine
from pokajan.core.state import EndReason, Phase
from pokajan.server.protocol import ActionResponse

pytestmark = pytest.mark.scenario


def rigged(
    rules,
    hands,
    *,
    coins=None,
    deck=(),
    current_seat=0,
    phase=Phase.AWAIT_DISCARD,
):
    """An engine placed in an exact position, bypassing the deal."""
    engine = Engine.new_game(rules, seed=1, bonus_character=rules.cards.char_index("c"))
    s = engine.state
    players = rules.play.players

    s.hands = [rules.cards.from_pairs(h) for h in hands]
    s.coins = list(coins) if coins is not None else [rules.play.initial_coins] * players
    s.deck = [rules.cards.slot(*card) for card in deck]
    s.table = rules.cards.zeros()
    s.scored = rules.cards.zeros()
    s.discards = [rules.cards.zeros() for _ in range(players)]
    s.recent_discards = [[] for _ in range(players)]
    s.calls_made = [0] * players
    s.coins_won = [0] * players
    s.coins_paid = [0] * players
    s.coins_minted = 0
    s.current_seat = current_seat
    s.turn_index = 0
    s.phase = phase
    s.finished = False
    s.end_reason = None
    s.last_discard_slot = None
    s.last_discard_seat = None
    s.chain_seat = None
    s.chain_needs_discard = False
    s.chain_from_claim = False
    s.claim_eligible = []
    s.claim_responses = {}
    return engine


def act(engine, seat, action):
    return ActionResponse(game_id=engine.state.game_id, seat=seat, action=action)


def discard(engine, seat, character, color):
    return act(engine, seat, engine.state.rules.cards.slot(character, color))


def call(engine, seat):
    return act(engine, seat, call_action(engine.state.rules))


def decline(engine, seat):
    return act(engine, seat, pass_action(engine.state.rules))


def seats_asked(engine):
    return sorted(r.seat for r in engine.pending_decisions())


# ------------------------------------------------------- bankruptcy floor ---

def test_bankrupt_payer_stops_at_zero_but_caller_is_paid_in_full(fixture_rules):
    """The confirmed asymmetry, and the only place coins are created.

    Seat 0 discards into seat 1's triple while holding just 10 coins. Seat 1 is
    owed 100; seat 0 can only find 10. Seat 1 receives all 100 anyway, so 90 coins
    enter the game that nobody paid.
    """
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("f", "pink")],
            [("a", "orange"), ("a", "pink")],
            [("e", "blue")],
            [("e", "orange")],
        ],
        coins=[10, 1000, 1000, 1000],
        deck=[("f", "blue")] * 10,
    )

    engine.submit(discard(engine, 0, "a", "blue"))
    assert seats_asked(engine) == [1]
    engine.submit(call(engine, 1))

    s = engine.state
    assert s.coins[0] == 0, "the payer floors at zero rather than going negative"
    assert s.coins[1] == 1100, "the caller still receives the full payout"
    assert s.coins_minted == 90
    assert sum(s.coins) == 3010 + s.coins_minted, "seat 0 started this position on 10"
    assert s.finished and s.end_reason == EndReason.BANKRUPT.value


def test_a_solvent_payment_mints_nothing(fixture_rules):
    """The control case: same hand, a payer who can afford it."""
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("f", "pink")],
            [("a", "orange"), ("a", "pink")],
            [("e", "blue")],
            [("e", "orange")],
        ],
        deck=[("f", "blue")] * 10,
    )
    engine.submit(discard(engine, 0, "a", "blue"))
    engine.submit(call(engine, 1))

    s = engine.state
    assert s.coins[0] == 900
    assert s.coins[1] == 1100
    assert s.coins_minted == 0
    assert sum(s.coins) == 4000


# --------------------------------------------------------------- payment ----

def test_claiming_bills_the_discarder_alone(fixture_rules):
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("f", "pink")],
            [("a", "orange"), ("a", "pink")],
            [("e", "blue")],
            [("e", "orange")],
        ],
        deck=[("f", "blue")] * 10,
    )
    engine.submit(discard(engine, 0, "a", "blue"))
    engine.submit(call(engine, 1))

    assert engine.state.coins == [900, 1100, 1000, 1000]


def test_self_drawn_hand_is_split_by_the_other_three(fixture_rules):
    """100 does not divide by 3, so the remainder has to land somewhere.

    Pinned deliberately: the real game's rounding here is unknown (TODO(M2)), and
    this test is what makes correcting it a conscious change.
    """
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("a", "orange"), ("a", "pink"), ("f", "blue")],
            [("e", "blue")],
            [("e", "orange")],
            [("e", "pink")],
        ],
        deck=[("f", "orange")] * 10,
        phase=Phase.AWAIT_IN_TURN_CALL,
    )
    engine.submit(call(engine, 0))

    s = engine.state
    assert s.coins[0] == 1100
    assert sum(1000 - c for c in s.coins[1:]) == 100
    assert s.coins[1:] == [966, 967, 967]
    assert s.coins_minted == 0


# ------------------------------------------------------- claim resolution ---

def test_the_bigger_payout_wins_regardless_of_seat_order(fixture_rules):
    """Seat 1 is earlier in order but seat 2's hand pays far more.

    Seat 1 claims a mixed triple of c: 100 x3 bonus = 300.
    Seat 2 claims the whole 'right' group in blue: 400 x2 mono x3 bonus = 2400.
    """
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("c", "blue"), ("f", "pink")],
            [("c", "orange"), ("c", "pink")],
            [("d", "blue"), ("e", "blue"), ("f", "blue")],
            [("a", "orange")],
        ],
        deck=[("a", "pink")] * 12,
    )
    engine.submit(discard(engine, 0, "c", "blue"))
    assert seats_asked(engine) == [1, 2]

    engine.submit([call(engine, 1), call(engine, 2)])

    s = engine.state
    assert s.coins[2] == 3400, "seat 2 should have won the card"
    assert s.coins[1] == 1000, "seat 1 gets nothing for losing the claim"
    assert s.calls_made[1] == 0 and s.calls_made[2] == 1


def test_ties_go_to_the_earliest_seat_after_the_discarder(fixture_rules):
    """Seats 1 and 3 hold identical claims; priority runs from the discarder."""
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("f", "pink")],
            [("a", "orange"), ("a", "pink")],
            [("e", "blue")],
            [("a", "orange"), ("a", "pink")],
        ],
        deck=[("f", "blue")] * 12,
    )
    engine.submit(discard(engine, 0, "a", "blue"))
    assert seats_asked(engine) == [1, 3]
    engine.submit([call(engine, 1), call(engine, 3)])

    assert engine.state.calls_made[1] == 1
    assert engine.state.calls_made[3] == 0


def test_priority_is_relative_to_the_discarder_not_to_seat_zero(fixture_rules):
    """The same tie, discarded by seat 2, now resolves the other way.

    Order from seat 2 is [3, 0, 1], so seat 3 takes it — which is what
    distinguishes "earliest in play order" from "lowest seat number".
    """
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("e", "blue")],
            [("a", "orange"), ("a", "pink")],
            [("a", "blue"), ("f", "pink")],
            [("a", "orange"), ("a", "pink")],
        ],
        deck=[("f", "blue")] * 12,
        current_seat=2,
    )
    engine.submit(discard(engine, 2, "a", "blue"))
    assert seats_asked(engine) == [1, 3]
    engine.submit([call(engine, 1), call(engine, 3)])

    assert engine.state.calls_made[3] == 1
    assert engine.state.calls_made[1] == 0


def test_a_declined_discard_stays_on_the_table_and_play_moves_on(fixture_rules):
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("f", "pink")],
            [("a", "orange"), ("a", "pink")],
            [("e", "blue")],
            [("e", "orange")],
        ],
        deck=[("f", "blue")] * 12,
    )
    slot = r.cards.slot("a", "blue")
    engine.submit(discard(engine, 0, "a", "blue"))
    engine.submit(decline(engine, 1))

    s = engine.state
    assert s.table[slot] == 1, "an unclaimed discard remains on the table"
    assert s.calls_made == [0, 0, 0, 0]
    assert s.current_seat == 1, "play continues to the next seat"


def test_only_seats_that_could_use_the_card_are_asked(fixture_rules):
    """Claiming is not offered as a no-op; seat 2 cannot use an 'a'."""
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("f", "pink")],
            [("a", "orange"), ("a", "pink")],
            [("e", "blue"), ("d", "blue")],
            [("e", "orange")],
        ],
        deck=[("f", "blue")] * 12,
    )
    engine.submit(discard(engine, 0, "a", "blue"))
    assert seats_asked(engine) == [1]


# ------------------------------------------------------------- turn flow ----

def test_an_out_of_turn_claim_neither_moves_the_turn_nor_forces_a_discard(fixture_rules):
    """Confirmed rule, and the reason claiming is so much better than self-drawing.

    Seat 2 claims seat 0's discard, scores, refills — and then play simply carries
    on to seat 1. Seat 2 pays no discard for the privilege.
    """
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("f", "pink")],
            [("e", "blue")],
            [("a", "orange"), ("a", "pink")],
            [("e", "orange")],
        ],
        deck=[("f", "blue"), ("e", "pink"), ("d", "orange"),
              ("f", "orange"), ("d", "pink"), ("b", "blue")],
    )
    engine.submit(discard(engine, 0, "a", "blue"))
    engine.submit(call(engine, 2))

    s = engine.state
    assert s.calls_made[2] == 1
    assert s.current_seat == 1, "the turn passes to seat 1, not to the claimer"
    assert seats_asked(engine) == [1]
    assert s.chain_seat is None


def test_an_in_turn_call_still_ends_with_a_discard(fixture_rules):
    """Under the current config flag — the one flow question still open (TODO M2)."""
    r = fixture_rules
    assert r.play.discard_after_in_turn_call is True

    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("a", "orange"), ("a", "pink"), ("f", "blue")],
            [("e", "blue")],
            [("e", "orange")],
            [("e", "pink")],
        ],
        deck=[("b", "blue"), ("d", "orange"), ("f", "orange"), ("d", "pink")],
        phase=Phase.AWAIT_IN_TURN_CALL,
    )
    engine.submit(call(engine, 0))

    s = engine.state
    assert s.phase is Phase.AWAIT_DISCARD
    assert s.current_seat == 0
    assert seats_asked(engine) == [0]


def test_a_made_hand_cannot_be_called_on_someone_elses_turn(fixture_rules):
    """You can sit on a complete hand and be unable to do anything about it.

    Seat 1 holds a finished triple while seat 0 is to act. Seat 1 is not asked
    anything, and is not offered a claim on a card it cannot use.
    """
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("f", "pink"), ("d", "blue")],
            [("a", "blue"), ("a", "orange"), ("a", "pink")],
            [("e", "blue")],
            [("e", "orange")],
        ],
        deck=[("b", "blue")] * 12,
    )
    assert seats_asked(engine) == [0]
    engine.submit(discard(engine, 0, "f", "pink"))

    # Nobody could use the 'f', so no claim window opened at all and play ran
    # straight on. Seat 1 is asked now, but as its own turn — not as a claim.
    s = engine.state
    assert s.current_seat == 1
    assert s.phase is Phase.AWAIT_IN_TURN_CALL
    pending = engine.pending_decisions()
    assert [r.seat for r in pending] == [1]
    assert pending[0].decision == "IN_TURN_CALL", (
        "a made hand must wait for your turn; it may never be called as a claim"
    )


def test_a_made_hand_does_not_make_every_discard_claimable(fixture_rules):
    """The sharp edge of the rule above.

    Seat 1 holds a completed triple of 'a'. Seat 0 discards a 'd', which does
    nothing for seat 1. Asking "would my hand score with this card added?" answers
    yes — the hand already scored — so a naive eligibility check hands seat 1 a
    free claim on a card it cannot use, and with it the ability to score a hand it
    was supposed to be waiting to call. Eligibility has to be "can I score a hand
    that *spends* this card".
    """
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("d", "pink"), ("b", "blue")],
            [("a", "blue"), ("a", "orange"), ("a", "pink")],
            [("e", "blue")],
            [("e", "orange")],
        ],
        deck=[("b", "orange")] * 12,
    )
    engine.submit(discard(engine, 0, "d", "pink"))

    assert engine.state.calls_made[1] == 0
    assert engine.state.current_seat == 1
    assert engine.pending_decisions()[0].decision == "IN_TURN_CALL"


# ---------------------------------------------------------------- chains ----

def test_a_refill_that_completes_another_hand_offers_a_second_call(fixture_rules):
    """The chain rule: score, refill to the limit, and you may go again."""
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("a", "orange"), ("a", "pink"), ("b", "blue"), ("b", "orange")],
            [("e", "blue")],
            [("e", "orange")],
            [("e", "pink")],
        ],
        # Drawn from the end: b-pink first, completing a second triple.
        deck=[("f", "blue"), ("d", "orange"), ("b", "pink")],
        phase=Phase.AWAIT_IN_TURN_CALL,
    )
    engine.submit(call(engine, 0))

    s = engine.state
    assert s.calls_made[0] == 1
    assert s.phase is Phase.AWAIT_CHAIN
    assert seats_asked(engine) == [0]

    engine.submit(call(engine, 0))
    assert engine.state.calls_made[0] == 2


def test_a_chain_may_be_declined(fixture_rules):
    """Chaining is optional, because every refill drains the shared deck."""
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("a", "orange"), ("a", "pink"), ("b", "blue"), ("b", "orange")],
            [("e", "blue")],
            [("e", "orange")],
            [("e", "pink")],
        ],
        deck=[("f", "blue"), ("d", "orange"), ("b", "pink")],
        phase=Phase.AWAIT_IN_TURN_CALL,
    )
    engine.submit(call(engine, 0))
    assert engine.state.phase is Phase.AWAIT_CHAIN

    engine.submit(decline(engine, 0))
    s = engine.state
    assert s.calls_made[0] == 1
    assert s.phase is Phase.AWAIT_DISCARD, "declining still owes the in-turn discard"


# ------------------------------------------------------------ deck empty ----

def test_running_the_deck_dry_during_a_refill_ends_the_game(fixture_rules):
    """A long chain can end the game on demand — a real strategic lever."""
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("a", "orange"), ("a", "pink"), ("b", "blue"), ("b", "orange")],
            [("e", "blue")],
            [("e", "orange")],
            [("e", "pink")],
        ],
        deck=[("f", "blue")],          # one card, but the refill needs three
        phase=Phase.AWAIT_IN_TURN_CALL,
    )
    engine.submit(call(engine, 0))

    s = engine.state
    assert s.finished
    assert s.end_reason == EndReason.DECK_EMPTY.value
    assert s.deck_remaining == 0
    assert s.calls_made[0] == 1, "the call still scored before the deck ran out"


def test_a_finished_game_asks_nothing_and_accepts_nothing(fixture_rules):
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("a", "orange"), ("a", "pink"), ("b", "blue"), ("b", "orange")],
            [("e", "blue")],
            [("e", "orange")],
            [("e", "pink")],
        ],
        deck=[("f", "blue")],
        phase=Phase.AWAIT_IN_TURN_CALL,
    )
    engine.submit(call(engine, 0))
    assert engine.pending_decisions() == []
    with pytest.raises(ValueError, match="finished"):
        engine.submit(decline(engine, 0))


# ------------------------------------------------------------ validation ----

def test_illegal_actions_are_rejected(fixture_rules):
    r = fixture_rules
    engine = rigged(
        r,
        hands=[
            [("a", "blue"), ("f", "pink")],
            [("e", "blue")],
            [("e", "orange")],
            [("e", "pink")],
        ],
        deck=[("b", "blue")] * 12,
    )
    with pytest.raises(ValueError, match="illegal action"):
        engine.submit(discard(engine, 0, "d", "pink"))   # not held
    with pytest.raises(ValueError, match="not asked to act"):
        engine.submit(discard(engine, 2, "e", "orange"))  # not their turn
