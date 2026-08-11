"""Turning a stream of frames into what has happened, without inventing any of it.

The reader's failure mode is a wrong card and it shows up in a log. The accumulator's is a
wrong *history*, which does not: a ledger that quietly splits one payout into two writes a
plausible line either way, and the round only looks wrong at the end, if at all. So most of
what is pinned here is refusal -- the cases where the right answer is to stop rather than to
produce a number.

Readings are synthetic and hand-built. The real logs in `data/games/` are records of online
matches against real people and are gitignored, so they are measured by `scripts/replay.py` on
the machine that holds them; what is checked here is the machinery, on inputs chosen to hit
the edges those rounds only happened to contain.
"""

from __future__ import annotations

import pytest

from pokajan.core.rules import HandKind, load_default
from pokajan.vision.accumulate import (
    BONUS_VOTE_MIN,
    LEDGER_PATIENCE,
    CoinLedger,
    DiscardFloor,
    RoundBelief,
    Shape,
    meld_shape,
    shapes_paying,
)
from pokajan.vision.reader import Card, FrameReading, Meld

pytestmark = pytest.mark.invariant

SEATS = ("bottom", "left", "top", "right")


@pytest.fixture(scope="module")
def rules():
    return load_default()


def reading(**kwargs) -> FrameReading:
    """A frame reading with everything the accumulator wants, overridable one field at a time."""
    base = dict(
        at=0.0,
        screen="table",
        deck_remaining=60,
        coins={},
        hand=tuple(Card("gawr_gura", "blue") for _ in range(7)),
        roster=("gen1", "gen4", "holox", "id1"),
        bonus="tokoyami_towa",
    )
    base.update(kwargs)
    return FrameReading(**base)


# --------------------------------------------------------------- the ledger ---
def test_a_payout_seen_all_at_once_is_read_off_the_coins(rules):
    ledger = CoinLedger(rules)
    event = ledger.observe({"bottom": 1180, "left": 820, "top": 1000, "right": 1000})

    assert event is not None
    assert (event.winner, event.amount) == ("bottom", 180)
    assert event.claimed and event.payer == "left"
    assert event.minted == 0
    assert ledger.balances


def test_a_payout_arriving_one_seat_at_a_time_is_still_one_payout(rules):
    """The case that broke the first design, and the reason the ledger waits.

    The four coin displays are not read on the same frame -- one refuses while its neighbour
    reads -- so a payout arrives in pieces. Settling on the first non-zero difference turns
    this single event into "bottom won 180 from nobody" followed by "left lost 180 to nobody",
    and the books never recover.
    """
    ledger = CoinLedger(rules)

    assert ledger.observe({"bottom": 1180}) is None      # half the story
    assert ledger.pending

    event = ledger.observe({"left": 820})
    assert event is not None
    assert (event.winner, event.amount, event.payer) == ("bottom", 180, "left")
    assert not ledger.pending


def test_a_self_drawn_hand_is_split_evenly_by_the_other_three(rules):
    ledger = CoinLedger(rules)
    event = ledger.observe({"left": 1120, "bottom": 960, "top": 960, "right": 960})

    assert event is not None
    assert not event.claimed and event.payer is None
    assert event.paid == {"bottom": 40, "top": 40, "right": 40}
    assert event.minted == 0


def test_the_bankruptcy_floor_mints_the_shortfall(rules):
    """The live 840 from `data/games/20260810T230853.jsonl`, records 55-57.

    Left called a monochrome triple with bottom holding 160. Bottom paid what it had and
    stopped at zero, left still received the whole 840, and the 120 difference was minted --
    which is the asymmetric floor, and the reason `sum(coins) == 4000 + minted` rather than
    `== 4000`.
    """
    ledger = CoinLedger(rules, coins={"bottom": 160, "left": 1000, "top": 610, "right": 2230})
    assert ledger.balances and ledger.minted == 0

    event = ledger.observe({"bottom": 0, "left": 1840, "top": 330, "right": 1950})
    assert event is not None
    assert event.amount == 840 and event.paid == {"bottom": 160, "top": 280, "right": 280}
    assert event.minted == 120
    assert ledger.total == 4120 and ledger.balances


def test_a_seat_falling_short_without_reaching_the_floor_is_refused(rules):
    """Underpaying is only ever explained by the floor, so away from it there is no story."""
    ledger = CoinLedger(rules)
    # Bottom pays 100 of the 280 it owes, but its display says 720 rather than 0.
    assert ledger.observe({"bottom": 900, "left": 1840, "top": 720, "right": 720}) is None
    assert ledger.pending


def test_the_coin_granularity_divides_everything_the_rules_can_pay(rules):
    """A theorem, not a heuristic, and derived rather than written as 10 because the payout
    table is the one thing here still expected to be corrected.

    Every payout divides by three, every three-way share is a multiple of ten, the stack starts
    at a multiple of ten, and the bankruptcy floor stops a payer at exactly zero. So no coin
    total can ever be anything else.
    """
    from pokajan.core.rules import HandKind
    from pokajan.vision.accumulate import _coin_granularity

    step = _coin_granularity(rules)
    assert step > 1

    for kind, size in ((HandKind.TRIPLE, None), (HandKind.GROUP, 3),
                       (HandKind.GROUP, 4), (HandKind.GROUP, 5)):
        for mono in (False, True):
            for bonus in (0, 1, 3):
                amount = rules.payout(kind, group_size=size, monochrome=mono,
                                      bonus_copies=bonus)
                if not amount:
                    continue
                assert amount % step == 0, f"{amount} is not a multiple of {step}"
                assert amount % 3 == 0, "a self-drawn hand splits three ways evenly"
                assert (amount // 3) % step == 0, f"share of {amount} breaks the step"
    assert rules.play.initial_coins % step == 0


def test_a_coin_reading_that_cannot_be_a_coin_total_is_thrown_out(rules):
    """A digit misread is otherwise indistinguishable from a real coin change, and one costs a
    whole round: a `bottom` seat read as 1 while the game-over banner came up put a 138-frame
    round out of balance by exactly one coin, and nothing else in it was wrong.

    Measured across every logged round, 4 readings of 8391 break the rule -- 1, 13, 4 and 1, all
    on covered or animating displays.
    """
    ledger = CoinLedger(rules)
    assert ledger.granularity == 10

    ledger.observe({"bottom": 1000, "left": 1000, "top": 1000, "right": 1000})
    ledger.observe({"bottom": 1})           # the game-over banner, misread

    assert ledger.latest["bottom"] == 1000, "the impossible reading must not land"
    assert ledger.rejected == {"bottom": 1}
    assert not ledger.pending and ledger.balances


def test_a_seat_genuinely_reaching_zero_is_not_thrown_out(rules):
    """Zero is a multiple of everything, and it is where the bankruptcy floor puts a payer. A
    rule that rejected it would refuse exactly the rounds this project cares most about."""
    ledger = CoinLedger(rules, coins={"bottom": 160, "left": 1000, "top": 610, "right": 2230})

    event = ledger.observe({"bottom": 0, "left": 1840, "top": 330, "right": 1950})

    assert event is not None and event.minted == 120
    assert ledger.latest["bottom"] == 0 and not ledger.rejected


def test_two_winners_at_once_is_two_payouts_not_a_bigger_one(rules):
    ledger = CoinLedger(rules)
    assert ledger.observe({"bottom": 1120, "left": 1120, "top": 880, "right": 880}) is None
    assert ledger.pending


def test_an_unexplainable_change_is_announced_rather_than_absorbed(rules):
    """The guard. Two payouts landing with the state between them never seen.

    The damage from being wrong here is silent, so the ledger stops and says so instead of
    accepting whichever legal-looking event the combined difference happens to resemble.
    """
    ledger = CoinLedger(rules)
    for _ in range(LEDGER_PATIENCE + 1):
        assert ledger.observe({"bottom": 1234, "left": 900, "top": 1000, "right": 866}) is None

    assert ledger.lost is not None
    assert "no legal payout explains it" in ledger.lost
    # And it stays lost: a later frame that would balance on its own does not undo it.
    assert ledger.observe({"bottom": 1180, "left": 820, "top": 1000, "right": 1000}) is None


def test_patience_is_not_spent_while_the_coins_agree(rules):
    ledger = CoinLedger(rules)
    for _ in range(LEDGER_PATIENCE * 3):
        assert ledger.observe({"bottom": 1000, "left": 1000, "top": 1000, "right": 1000}) is None
    assert ledger.lost is None


# ---------------------------------------------------- payout amount -> shape ---
def test_an_amount_names_the_shapes_that_could_have_paid_it(rules):
    assert [str(s) for s in shapes_paying(120, rules)] == ["triple"]
    assert [str(s) for s in shapes_paying(180, rules, group_sizes=[3, 4])] == ["3-group"]


def test_a_triple_carries_three_bonus_copies_or_none(rules):
    """What made 930 decode uniquely, and it is a fact about the cards rather than a heuristic.

    A triple's scoring set is three copies of one holomem, so either all three are the bonus
    holomem or none is. Allowing one or two would put a phantom `120 + 90` alongside the real
    `180`, and every three-group payout would become ambiguous with a triple.
    """
    assert {s.bonus_copies for s in shapes_paying(390, rules) if s.kind is HandKind.TRIPLE} == {3}
    assert shapes_paying(210, rules) == ()          # 120 + 90 pays nothing

    # The confirmed real observation: 840 + 90 has exactly one explanation.
    assert [str(s) for s in shapes_paying(930, rules)] == ["mono 4-group +1 bonus"]


def test_a_roster_without_five_member_groups_narrows_an_ambiguous_amount(rules):
    """480 is a five-group or a monochrome three-group, until the table says which exist."""
    assert len(shapes_paying(480, rules, group_sizes=[3, 4, 5])) == 2
    assert [str(s) for s in shapes_paying(480, rules, group_sizes=[3, 4])] == ["mono 3-group"]


def test_an_amount_no_hand_pays_is_reported_as_such(rules):
    assert shapes_paying(137, rules) == ()


# ------------------------------------------------------------ discard floor ---
def test_the_floor_is_the_most_ever_seen_at_once():
    """Merged by `max`, which needs no alignment between overlapping views of a growing row.

    Sequence alignment was the alternative and it is ambiguous exactly where it matters: a row
    reading `watame, watame, gura` overlaps a later `watame, gura, polka` two different ways,
    and picking the wrong one puts a card into `table` that nothing ever saw.
    """
    floor = DiscardFloor()
    floor.observe("bottom", [Card("tsunomaki_watame", "pink"), Card("tsunomaki_watame", "blue")])
    floor.observe("bottom", [Card("tsunomaki_watame", "blue"), Card("gawr_gura", "blue")])

    # One pink and one blue watame were seen together, so both are on the table. The second
    # view repeats the blue one; it is not a third.
    assert floor.counts() == {
        "tsunomaki_watame:pink": 1, "tsunomaki_watame:blue": 1, "gawr_gura:blue": 1,
    }


def test_a_half_read_card_is_dropped_rather_than_recorded_as_a_partial():
    """A count vector cannot express "some pink card", and guessing the other half puts a card
    the belief will treat as certain into a slot nothing ever saw."""
    floor = DiscardFloor()
    floor.observe("bottom", [Card("gawr_gura", None), Card(None, "blue"), Card("omaru_polka", "pink")])

    assert floor.counts() == {"omaru_polka:pink": 1}


def test_the_seats_that_cannot_be_read_are_named_from_the_start():
    """The two side seats lay their discards out as diagonal staircases, so the floor is short
    by an amount it must declare rather than imply."""
    floor = DiscardFloor()
    assert floor.unread == {"left", "right"}

    floor.observe("bottom", [Card("gawr_gura", "blue")])
    assert floor.unread == {"left", "right"}


# -------------------------------------------------------------- the meld ---
GEN1 = {"amane_kanata": "gen1", "aki_rosenthal": "gen1",
        "akai_haato": "gen1", "yozora_mel": "gen1"}
SIZES = {"gen1": 4}


def meld(*pairs) -> tuple[Card, ...]:
    return tuple(Card(who, colour) for who, colour in pairs)


def test_three_of_one_holomem_is_a_triple():
    shape = meld_shape(meld(("aki_rosenthal", "blue"), ("aki_rosenthal", "orange"),
                            ("aki_rosenthal", "blue")))
    assert shape == Shape(HandKind.TRIPLE, None, False, 0)


def test_one_colour_throughout_is_monochrome():
    shape = meld_shape(meld(("aki_rosenthal", "blue"), ("aki_rosenthal", "blue"),
                            ("aki_rosenthal", "blue")))
    assert shape is not None and shape.monochrome


def test_a_triple_of_the_bonus_holomem_carries_three_copies_not_one():
    shape = meld_shape(meld(("aki_rosenthal", "blue"), ("aki_rosenthal", "pink"),
                            ("aki_rosenthal", "blue")), bonus="aki_rosenthal")
    assert shape is not None and shape.bonus_copies == 3


def test_one_of_each_member_is_that_group():
    shape = meld_shape(meld(("amane_kanata", "blue"), ("aki_rosenthal", "pink"),
                            ("akai_haato", "blue"), ("yozora_mel", "orange")),
                       group_of=GEN1, group_size=SIZES)
    assert shape == Shape(HandKind.GROUP, 4, False, 0)


def test_an_incomplete_group_is_not_a_shape():
    """Three of a four-member group is not a hand, and calling it a 3-group would invent a
    payout the table has no row for."""
    assert meld_shape(meld(("amane_kanata", "blue"), ("aki_rosenthal", "pink"),
                           ("akai_haato", "blue")),
                      group_of=GEN1, group_size=SIZES) is None


def test_a_half_read_meld_describes_nothing():
    assert meld_shape(meld(("aki_rosenthal", "blue"), (None, "pink"),
                           ("aki_rosenthal", "blue"))) is None
    assert meld_shape(meld(("aki_rosenthal", "blue"), ("aki_rosenthal", None),
                           ("aki_rosenthal", "blue"))) is None


def test_two_of_one_and_one_of_another_is_neither_shape():
    assert meld_shape(meld(("aki_rosenthal", "blue"), ("aki_rosenthal", "pink"),
                           ("amane_kanata", "blue")),
                      group_of=GEN1, group_size=SIZES) is None


def test_a_read_meld_collapses_scored_from_a_range_to_cards(rules):
    """The whole point of reading the meld. 840 is a monochrome triple or a monochrome
    four-group, and the coins cannot tell which; the cards can."""
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))
    assert belief.scored_span == (0, 0)

    # Not the bonus holomem, which the fixture makes tokoyami_towa: a monochrome triple of
    # *that* pays 840 + 3x90 and is a different event. See the test below.
    payout = belief.observe(reading(
        deck_remaining=60, melds=(Meld('bottom', meld(*[("gawr_gura", "pink")] * 3)),),
        coins={"bottom": 1840, "left": 1000, "top": 1000, "right": 160}))

    assert payout is not None and payout.meld
    assert [str(shape) for shape in payout.shapes] == ["mono triple"]
    assert belief.scored_cards == 3          # was ambiguous between 3 and 4
    assert belief.scored == {"gawr_gura:pink": 3}


def test_a_monochrome_triple_of_the_bonus_holomem_pays_the_full_three_copies(rules):
    """840 + 3x90, not 840 + 90. The scoring set is three copies, so all three are bonus.

    Worth its own test because getting it wrong is invisible in the ledger and quiet in the
    reader: the meld would simply fail to confirm, `scored` would stay a range, and nothing
    would say why.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))
    payout = belief.observe(reading(
        deck_remaining=60, melds=(Meld('bottom', meld(*[("tokoyami_towa", "pink")] * 3)),),
        # Claimed off right, which holds 1000 and cannot cover 1110: it floors at 0 and the
        # 110 shortfall is minted, so the totals come to 4110.
        coins={"bottom": 2110, "left": 1000, "top": 1000, "right": 0}))

    assert payout is not None and payout.amount == 1110
    assert payout.minted == 110 and belief.ledger.total == 4110 and belief.ledger.balances
    assert [str(shape) for shape in payout.shapes] == ["mono triple +3 bonus"]
    assert belief.scored == {"tokoyami_towa:pink": 3}


def test_a_meld_the_amount_forbids_is_not_attached(rules):
    """Disagreement attaches nothing rather than overruling the amount.

    The coin displays are the better-checked channel -- four rules cross-check every event --
    and the meld comes out of a box verified on two frames. So a mismatch downgrades the meld,
    never the ledger.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))
    # A plain triple pays 120, not 480.
    payout = belief.observe(reading(
        deck_remaining=60,
        melds=(Meld('bottom', meld(("gawr_gura", "blue"), ("gawr_gura", "pink"), ("gawr_gura", "orange"))),),
        coins={"bottom": 1480, "left": 840, "top": 840, "right": 840}))

    assert payout is not None and payout.meld == ()
    assert len(payout.shapes) > 1            # still the range the amount permits
    assert belief.scored == {}


def test_a_meld_seen_a_frame_before_the_coins_settle_still_counts(rules):
    """The two channels are not in step: the meld is drawn while the payout animates and the
    coins finish moving a frame or two later."""
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))
    belief.observe(reading(deck_remaining=60, coins={},
                           melds=(Meld("bottom", meld(("gawr_gura", "blue"),
                                                      ("gawr_gura", "pink"),
                                                      ("gawr_gura", "orange"))),)))
    payout = belief.observe(reading(
        deck_remaining=60,
        coins={"bottom": 1120, "left": 960, "top": 960, "right": 960}))

    assert payout is not None and len(payout.meld) == 3
    assert belief.scored == {"gawr_gura:blue": 1, "gawr_gura:pink": 1, "gawr_gura:orange": 1}


def test_a_stale_meld_is_not_reused_on_the_next_payout(rules):
    """Held only until it is spent. Otherwise one meld would confirm every later payout whose
    amount happened to permit its shape, and `scored` would fill with cards seen once."""
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))
    belief.observe(reading(
        deck_remaining=60,
        melds=(Meld('bottom', meld(("gawr_gura", "blue"), ("gawr_gura", "pink"), ("gawr_gura", "orange"))),),
        coins={"bottom": 1120, "left": 960, "top": 960, "right": 960}))
    second = belief.observe(reading(
        deck_remaining=58,
        coins={"bottom": 1240, "left": 920, "top": 920, "right": 920}))

    assert second is not None and second.meld == ()
    assert sum(belief.scored.values()) == 3


# ------------------------------------------------------------ the whole round ---
def test_settled_facts_are_defended_rather_than_overwritten(rules):
    """A roster does not change inside a round, so a frame disagreeing is a misread."""
    belief = RoundBelief(rules)
    belief.observe(reading())
    belief.observe(reading(roster=("gen0", "gen2", "myth", "id2")))

    assert belief.roster == ("gen1", "gen4", "holox", "id1")
    assert belief.conflicts and "having settled on" in belief.conflicts[0]
    assert not belief.tracking.ready


def test_a_round_with_everything_read_is_ready_to_advise_on(rules):
    belief = RoundBelief(rules)
    belief.observe(reading(coins={s: 1000 for s in SEATS}))

    tracking = belief.tracking
    assert tracking.ready, tracking.reasons
    # And it never claims the table is complete, because it is not.
    assert not tracking.table_complete
    assert any("diagonal staircases" in reason for reason in tracking.table_reasons)


@pytest.mark.parametrize("missing, expect", [
    ({"roster": None}, "roster"),
    ({"bonus": None}, "bonus holomem"),
    ({"deck_remaining": None}, "deck counter"),
    ({"hand": ()}, "hand at all"),
])
def test_anything_unread_blocks_advice_by_name(rules, missing, expect):
    belief = RoundBelief(rules)
    belief.observe(reading(coins={s: 1000 for s in SEATS}, **missing))

    tracking = belief.tracking
    assert not tracking.ready
    assert any(expect in reason for reason in tracking.reasons), tracking.reasons


def test_a_hand_with_a_card_or_two_unread_still_advises_and_counts_them(rules):
    """The bar used to be the *whole* hand on one frame, and that is seven chances to fail
    rather than one. A 34-frame round read every slot at some point and never all seven at
    once, so it produced nothing at all.

    The cost is real and is the reason this is capped rather than removed: a hand with a
    refused card is **not a smaller hand**, but the count vector built from it is, so the
    advisor prices a seven-card hand as six. `hand_unread` is what carries that through to the
    panel instead of leaving it implied.
    """
    belief = RoundBelief(rules)
    hand = (Card("gawr_gura", "blue"), Card(None, "pink")) + tuple(
        Card("tokoyami_towa", "blue") for _ in range(5))
    belief.observe(reading(coins={s: 1000 for s in SEATS}, hand=hand))

    assert belief.hand_unread == 1
    assert belief.tracking.ready, belief.tracking.reasons


def test_a_hand_mostly_unread_is_refused_rather_than_guessed_at(rules):
    """Past a couple of cards it stops being a caveat and starts being a guess."""
    belief = RoundBelief(rules)
    hand = (Card("gawr_gura", "blue"),) + tuple(Card(None, "pink") for _ in range(6))
    belief.observe(reading(coins={s: 1000 for s in SEATS}, hand=hand))

    assert belief.hand_unread == 6
    assert not belief.tracking.ready
    assert any("unread" in reason for reason in belief.tracking.reasons)


def test_the_hand_is_assembled_across_frames_not_demanded_from_one(rules):
    """Seven cards each refusing independently means a complete frame is rare, but the hand
    does not change between draws — so what one frame missed the next can supply."""
    belief = RoundBelief(rules)
    rest = tuple(Card("tokoyami_towa", "blue") for _ in range(5))
    belief.observe(reading(hand=(Card("gawr_gura", "blue"), Card(None, "pink")) + rest))
    assert belief.hand_unread == 1

    belief.observe(reading(hand=(Card(None, "blue"), Card("amane_kanata", "pink")) + rest))

    assert belief.hand_unread == 0
    assert [str(card) for card in belief.hand[:2]] == ["gawr_gura:blue", "amane_kanata:pink"]


def test_a_hand_that_disagrees_with_the_candidate_replaces_it(rules):
    """The check that makes merging safe. The hand re-sorts when a card is inserted and every
    card after it shifts along — measured, 3.85% of shared positions disagree between
    consecutive frames with the deck counter unmoved. Blending those would invent a card that
    was never held, so a disagreement throws the candidate away instead."""
    belief = RoundBelief(rules)
    rest = tuple(Card("tokoyami_towa", "blue") for _ in range(5))
    belief.observe(reading(hand=(Card("gawr_gura", "blue"), Card(None, "pink")) + rest))

    belief.observe(reading(hand=(Card("amane_kanata", "blue"), Card(None, "pink")) + rest))

    assert str(belief.hand[0]) == "amane_kanata:blue", "the newer frame wins outright"
    assert belief.hand_unread == 1, "and nothing was carried over from the old candidate"


def test_a_draw_drops_the_candidate_because_the_hand_has_changed(rules):
    """The deck counter falling is a draw, and a draw is a different hand. Merging across it
    would splice two positions together."""
    belief = RoundBelief(rules)
    rest = tuple(Card("tokoyami_towa", "blue") for _ in range(5))
    belief.observe(reading(deck_remaining=60,
                           hand=(Card("gawr_gura", "blue"), Card(None, "pink")) + rest))

    belief.observe(reading(deck_remaining=59,
                           hand=(Card(None, "blue"), Card("amane_kanata", "pink")) + rest))

    assert belief.hand_unread == 1, "the new frame stands alone"
    assert not belief.hand[0].known


def test_the_table_count_falls_out_of_conservation_without_reading_it(rules):
    """Every card is somewhere, so the deck counter bounds the table on its own.

    100 - 60 in the deck - 28 in hands leaves 12 unaccounted for, and one seat may be holding
    a drawn card. With nothing scored yet the table holds 11 or 12.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))

    assert belief.table_span == (11, 12)


def test_a_call_whose_refill_has_not_shown_yet_widens_the_table_downward(rules):
    """A self-drawn triple with the deck counter **unmoved**, which is a specific state and not
    a shortcut: the three cards have left the hand and the deck has not yet replaced them.

    So the caller is holding four, not seven, and `rest` still counts those three as outside
    the deck — they are in `scored` now. The table is unchanged at 12, and the old answer of 9
    came from assuming full hands, which the deck reading here rules out. Both halves move
    together once the refill lands: the deck falls by three, `rest` rises by three, hands go
    back to the limit, and the table comes out at 12 either way.

    Two real rounds were lost to this, both ending on a four-card call that bankrupted a seat —
    where the round stops and the refill never comes at all.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))
    # A plain triple: three cards, unambiguously.
    belief.observe(reading(deck_remaining=60,
                           coins={"bottom": 1120, "left": 960, "top": 960, "right": 960}))

    assert belief.scored_cards == 3
    assert belief.table_span == (8, 12)


def test_only_the_most_recent_call_can_still_be_waiting_on_its_refill(rules):
    """Otherwise every call in the round accumulates slack and the table bound goes soft.

    A chain has to refill to the limit before another call is legal — you cannot score out of a
    hand you have not got — so there is never a second unrefilled call queued behind the first.

    The order matters and a first attempt at this test missed the point: two calls of the same
    size cannot tell the two readings apart, because the slack is a `max` and not a sum. So the
    **bigger** call comes first here — a four-group, then a triple. Only the triple can still be
    waiting, so the slack is three, not the four the earlier call would allow.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))
    # 300 is a four-group and nothing else; 120 is a plain triple.
    belief.observe(reading(deck_remaining=60,
                           coins={"bottom": 1300, "left": 900, "top": 900, "right": 900}))
    belief.observe(reading(deck_remaining=60,
                           coins={"bottom": 1260, "left": 860, "top": 1020, "right": 860}))

    assert belief.scored_cards == 7, "a four-group then a triple"
    # rest 40, seven scored, and only the triple's caller still short: 40 - (28 - 3) - 7.
    assert belief.table_span[1] == 8


def test_an_ambiguous_payout_widens_the_table_rather_than_picking_a_shape(rules):
    """840 is a monochrome triple or a monochrome four-group, so the count is one card wide."""
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={s: 1000 for s in SEATS}))
    belief.observe(reading(deck_remaining=60,
                           coins={"bottom": 1840, "left": 1000, "top": 1000, "right": 160}))

    assert belief.scored_cards is None
    assert belief.scored_span == (3, 4)
    # Low end from four cards scored against a full hand plus a drawn one; high end from three
    # scored with the caller still four short, since the deck has not moved to refill it.
    assert belief.table_span == (7, 13)


def test_a_claimed_call_takes_its_card_off_the_discard_floor(rules):
    """`DiscardFloor` merges by `max` and never removes anything, which is right — it cannot
    tell a card that was claimed from one buried under a later discard. But a claimed call
    takes its card **off a pile**, so the floor goes on counting it as sitting on the table
    while `scored` counts it too, and the same card is in two places at once.

    Measured across the logged rounds: 42 cards in 15 rounds appear in both the floor and a
    meld that scored. Which card of the meld was the claimed one does not matter, because this
    bounds the count and not the identities.
    """
    belief = RoundBelief(rules)
    thrown = tuple(Card(name, "blue") for name in ("gawr_gura", "mori_calliope", "amelia_watson"))
    belief.observe(reading(coins={s: 1000 for s in SEATS},
                           discards={"bottom": thrown}))
    assert belief.discards.cards == 3 and belief.table_floor == 3

    # A triple claimed off `left`: that seat alone pays, so one card left a pile.
    belief.observe(reading(coins={"bottom": 1120, "left": 880, "top": 1000, "right": 1000},
                           discards={"bottom": thrown}))

    assert [p.claimed for p in belief.payouts] == [True]
    assert belief.discards.cards == 3, "the floor itself must stay monotone"
    assert belief.table_floor == 2


def test_an_unclaimed_call_leaves_the_discard_floor_alone(rules):
    """The other half, and the reason this is not just a fudge factor. A self-drawn call takes
    every card from the caller's own hand, so nothing leaves a pile and the floor still stands.
    """
    belief = RoundBelief(rules)
    thrown = tuple(Card(name, "blue") for name in ("gawr_gura", "mori_calliope", "amelia_watson"))
    belief.observe(reading(coins={s: 1000 for s in SEATS}, discards={"bottom": thrown}))

    # Split three ways, so nobody was claimed off.
    belief.observe(reading(coins={"bottom": 1120, "left": 960, "top": 960, "right": 960},
                           discards={"bottom": thrown}))

    assert [p.claimed for p in belief.payouts] == [False]
    assert belief.table_floor == 3


def test_conservation_refuses_to_answer_once_the_books_stop_balancing(rules):
    """A missed payout makes `scored` too small and this too large by the same amount.

    Left unguarded it would report a roomy table and wave through the very misread the
    cross-check below exists to catch. The six-frame log in `data/games/` is this case: it
    joins a round in progress, never sees the payouts behind the coin totals it can read.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(deck_remaining=60, coins={"bottom": 1760, "left": 1000,
                                                    "top": 520, "right": 800}))

    assert belief.ledger.pending and not belief.ledger.balances
    assert belief.table_span is None
    assert not belief.tracking.ready


def test_a_discard_floor_deeper_than_the_table_is_caught(rules):
    """Two independent channels disagreeing about the same quantity.

    Conservation rests on the deck counter and the payout amounts; the floor rests on
    segmenting a row of cards. When they disagree the row was misread -- most often segmented
    into more cards than it holds -- and that is invisible in the log without this check.
    """
    belief = RoundBelief(rules)
    # Deck at 71 with nothing scored leaves at most one card on the table.
    belief.observe(reading(deck_remaining=71, coins={s: 1000 for s in SEATS},
                           discards={"bottom": tuple(Card("gawr_gura", "blue")
                                                     for _ in range(4))}))

    assert belief.table_span == (0, 1)
    assert not belief.tracking.ready
    assert any("conservation allows at most" in reason for reason in belief.tracking.reasons)


# --------------------------------------------------------- the bonus by vote ---
#
# A live round refused the bonus on all 86 frames: `amelia_watson` at 0.42 against a 0.45
# floor, on 68 of them, with the other 18 scattered one apiece across holomem that were not
# even in the roster. Pale blonde art on a pale ground gives cross-correlation little to grip,
# so a correct read can sit under the floor for a whole round -- and without a way through, the
# round is unadvisable from its first frame to its last.

def voted(belief, who, times, **kw):
    for _ in range(times):
        belief.observe(reading(bonus=None, bonus_best=who, **kw))


def test_a_round_that_agrees_on_the_bonus_names_it(rules):
    belief = RoundBelief(rules)
    voted(belief, "tokoyami_towa", BONUS_VOTE_MIN * 2)

    assert belief.bonus is None                     # no single frame ever named it
    assert belief.bonus_by_vote == "tokoyami_towa"
    assert belief.bonus_holomem == "tokoyami_towa"


def test_too_few_frames_is_not_agreement(rules):
    belief = RoundBelief(rules)
    voted(belief, "tokoyami_towa", BONUS_VOTE_MIN - 1)

    assert belief.bonus_by_vote is None


def test_a_close_race_is_not_agreement(rules):
    """Noise scatters; it does not run neck and neck. A near-tie is two plausible readings of
    the same card, which is exactly when guessing mis-prices every hand in the round."""
    belief = RoundBelief(rules)
    voted(belief, "tokoyami_towa", BONUS_VOTE_MIN * 2)
    voted(belief, "gawr_gura", BONUS_VOTE_MIN)

    assert belief.bonus_by_vote is None


def test_a_holomem_this_round_cannot_deal_never_wins_the_vote(rules):
    """The roster is the cheapest possible check on it, and it is free."""
    belief = RoundBelief(rules)
    voted(belief, "gawr_gura", BONUS_VOTE_MIN * 5)   # myth, and the fixture deals none of it

    assert "gawr_gura" not in belief._group_of()
    assert belief.bonus_by_vote is None
    assert not belief.tracking.ready


def test_a_read_bonus_beats_a_vote(rules):
    """A vote is a fallback for when nothing could name it, not a second opinion.

    The challenger here has to be a holomem the round *can* deal — `amane_kanata` is Gen4 and
    the fixture deals Gen4 — or the roster check throws it out first and this passes without
    ever exercising the rule it is named for.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(coins={s: 1000 for s in SEATS}))    # names tokoyami_towa outright
    voted(belief, "amane_kanata", BONUS_VOTE_MIN * 5)

    assert "amane_kanata" in belief._group_of()
    assert belief.bonus == "tokoyami_towa"
    assert belief.bonus_holomem == "tokoyami_towa"
    # Asserted on the answer, not on `bonus_by_vote` returning None. It used to short-circuit
    # whenever an outright read existed, and that was a bug rather than a safeguard: settling a
    # disagreement between two outright reads has to consult the tally, and the short-circuit
    # made it silent in exactly that case. The tally may now say `amane_kanata` here; what
    # matters is that nothing asks it, because the outright reads do not disagree.
    assert len(belief.bonus_reads) == 1


def test_one_misread_of_the_bonus_does_not_outvote_the_round(rules):
    """The card does not change during a round, so two disagreeing outright reads mean one of
    them is wrong — and the round's own rankings know which.

    Latching on the first read and calling the second a conflict is the obvious spelling and it
    cost a 141-frame round: the bonus was read outright exactly twice, once `himemori_luna` and
    once `tokoyami_towa`, while 98 frames ranked `himemori_luna` top against 9. One bad frame
    beat 98 good ones and the round went unadvisable from that moment on.

    The misread deliberately arrives **first**. With the good read first, a latch still lands on
    the right answer and the test proves nothing — which is exactly what a first draft of this
    did. Order is the thing a latch is sensitive to, so order is what has to be wrong.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(coins={s: 1000 for s in SEATS},           # a single misread, first
                           bonus="amane_kanata", bonus_best="amane_kanata"))
    voted(belief, "tokoyami_towa", BONUS_VOTE_MIN * 5)
    belief.observe(reading(coins={s: 1000 for s in SEATS}))          # tokoyami_towa outright

    assert set(belief.bonus_reads) == {"tokoyami_towa", "amane_kanata"}
    assert belief.bonus == "tokoyami_towa"
    assert belief.tracking.ready, belief.tracking.reasons


def test_two_bonus_reads_the_round_cannot_separate_stay_unresolved(rules):
    """The other side of it, and the reason this is not just "believe the majority".

    With one read each and no body of rankings behind either, there is nothing to choose
    between them — so the bonus stays unread and the round says so, rather than picking the
    earlier one because it happened to arrive first.
    """
    belief = RoundBelief(rules)
    belief.observe(reading(coins={s: 1000 for s in SEATS}))          # tokoyami_towa outright
    belief.observe(reading(coins={s: 1000 for s in SEATS},
                           bonus="amane_kanata", bonus_best="amane_kanata"))

    assert belief.bonus is None
    assert not belief.tracking.ready
    assert any("do not settle which" in reason for reason in belief.tracking.reasons)


def test_a_bonus_the_round_ranks_top_but_never_read_does_not_break_a_tie(rules):
    """A vote landing on a holomem *nobody* read outright is two channels disagreeing, not one
    of them being noisy, and it must not resolve a tie between the two that were read."""
    belief = RoundBelief(rules)
    belief.observe(reading(coins={s: 1000 for s in SEATS}))          # tokoyami_towa outright
    belief.observe(reading(coins={s: 1000 for s in SEATS},
                           bonus="amane_kanata", bonus_best="amane_kanata"))
    voted(belief, "himemori_luna", BONUS_VOTE_MIN * 5)

    assert belief.bonus_by_vote == "himemori_luna"
    assert belief.bonus is None, "the tie-breaker must be one of the cards actually read"


def test_agreement_unblocks_a_round_nothing_could_read(rules):
    belief = RoundBelief(rules)
    voted(belief, "tokoyami_towa", BONUS_VOTE_MIN * 2,
          coins={s: 1000 for s in SEATS})

    tracking = belief.tracking
    assert tracking.ready, tracking.reasons
