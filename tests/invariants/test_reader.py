"""Reading one frame, and remembering it once the frame is gone.

The frame is the thing that does not survive: it is grabbed, read and dropped, so whatever
`FrameReading` fails to notice is lost for good and whatever `Journal` fails to write never
happened. Both halves are guarded here.

Everything is synthetic and the catalogues are injected, which is more than a convenience.
The behaviour worth pinning is what happens when a read *fails*, and an empty catalogue or a
stub that returns a silly number reaches that far more directly than a screenshot would.

The rule under all of it: **a field is either read or explained.** Never silently defaulted.
An advisor holding a wrong coin total gives confident advice about a different game, so a
caller has to be able to distinguish "seat two has 780" from "seat two could not be read"
without inspecting sentinels.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokajan.vision import layout
from pokajan.vision.digits import Number
from pokajan.vision.journal import NEW_DEAL_RISE, Journal, Record, RoundTracker
from pokajan.vision.reader import (
    DECK_BEFORE_DEAL,
    MAX_DECK_REMAINING,
    Card,
    FrameReading,
    TableReader,
)
from pokajan.vision.templates import TemplateSet, prepare

pytestmark = pytest.mark.invariant

FELT = (34, 110, 34)


def felt(width=1920, height=1080):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :] = FELT
    return pixels, layout.PlayArea(x=0, y=0, width=width, height=height)


class FixedDigits:
    """A digit reader that always says the same thing, so the checks around it can be tested."""

    def __init__(self, value: int | None, reason: str = "stub refusal") -> None:
        self._value = value
        self._reason = reason

    def __len__(self) -> int:
        return 10

    def read(self, region, *, max_digits: int = 6) -> Number:
        if self._value is None:
            return Number(None, "", 0.0, 0.0, self._reason)
        return Number(self._value, str(self._value), 0.9, 0.4)


def reader(*, digits=None) -> TableReader:
    """A reader with nothing loaded, so every read has to refuse rather than invent."""
    return TableReader(cards=None, digits=digits or FixedDigits(None),
                       templates=TemplateSet({}))


# ------------------------------------------------------------- the reading ----

def test_nothing_readable_means_everything_explained():
    """The core rule. Absent fields must each carry a reason, not a default."""
    pixels, area = felt()

    reading = reader().read(pixels, area)

    assert reading.deck_remaining is None
    assert reading.coins == {} and reading.roster is None and reading.bonus is None
    assert "deck_remaining" in reading.refusals
    assert "roster" in reading.refusals
    for seat in layout.SEAT_ORDER:
        assert f"coins_{seat}" in reading.refusals


def test_a_covered_panel_is_still_a_table():
    """Abandoning these frames would be the worst possible choice: a mid-payout frame is the
    only place a payout is observable, and a call's meld is the one moment `scored` can be
    seen at all before those cards leave the table for good."""
    pixels, area = felt()

    reading = reader().read(pixels, area)

    assert reading.screen == "table"
    assert "table:" in reading.refusals["roster"]
    assert "reveal:" in reading.refusals["roster"]


def test_a_deck_count_larger_than_the_deck_is_refused():
    """Refusing a *plausible* number is the job: a wrong deck count moves every belief in the
    game, and 140 looks no more alarming than 71 in a log."""
    pixels, area = felt()

    reading = reader(digits=FixedDigits(MAX_DECK_REMAINING + 40)).read(pixels, area)

    assert reading.deck_remaining is None
    assert str(MAX_DECK_REMAINING) in reading.refusals["deck_remaining"]


@pytest.mark.parametrize("value", [0, 30, DECK_BEFORE_DEAL, MAX_DECK_REMAINING])
def test_a_full_deck_is_a_real_reading(value):
    """This bound was 72 -- 100 less the 28-card opening deal -- and that was wrong.

    A live capture read 100 and had it thrown away, because the counter is on screen *before*
    the deal, showing the whole deck. The bound had been reasoned from arithmetic about a state
    the reader had never actually watched. That is the mild version of this project's usual
    failure, but the same mistake: preferring a derivation to an observation.
    """
    pixels, area = felt()

    reading = reader(digits=FixedDigits(value)).read(pixels, area)

    assert reading.deck_remaining == value
    assert "deck_remaining" not in reading.refusals


def test_a_partial_coin_read_has_no_total():
    """A sum of three seats would fail the 4000-plus-minted check and look like a rules
    problem rather than a missing read, which is a much more expensive thing to debug."""
    assert FrameReading(at=0.0, screen="table", coins={"bottom": 1000}).coins_total is None

    whole = FrameReading(at=0.0, screen="table",
                         coins={seat: 1000 for seat in layout.SEAT_ORDER})

    assert whole.coins_total == 4000


def test_a_card_is_known_only_when_both_halves_are():
    """Colour comes from the frame and the holomem from the artwork, and either can refuse."""
    assert Card("gawr_gura", "pink").known
    assert not Card("gawr_gura", None).known
    assert not Card(None, "pink").known
    assert str(Card(None, "pink")) == "?:pink"


# -------------------------------------------------------------- the journal ----

def test_a_record_survives_the_round_trip(tmp_path):
    journal = Journal(tmp_path / "session.jsonl")
    reading = FrameReading(
        at=1.5, screen="table", deck_remaining=46,
        coins={"bottom": 1610, "left": 1240, "top": 800, "right": 350},
        hand=(Card("gawr_gura", "pink"), Card(None, "blue")),
        roster=("gen1", "gen4", "id3", "myth"), bonus="gawr_gura",
        refusals={"hand_1": "too close to call"},
    )

    journal.append(Record.of(reading, round_index=2, crops={"discards_top": "t_top.png"}))
    (back,) = journal.records()

    assert back.round_index == 2 and back.deck_remaining == 46
    assert back.coins["right"] == 350
    assert back.hand == ["gawr_gura:pink", "?:blue"]
    assert back.roster == ["gen1", "gen4", "id3", "myth"]
    assert back.refusals == {"hand_1": "too close to call"}
    assert back.crops == {"discards_top": "t_top.png"}


def test_refusals_are_written_down(tmp_path):
    """A history with holes is far better than one with silent gaps: the holes are what tell a
    later reader that a turn was missed rather than that nothing happened."""
    journal = Journal(tmp_path / "session.jsonl")
    journal.append(Record.of(
        FrameReading(at=0.0, screen="table", refusals={"roster": "panel covered"}),
        round_index=0))

    (back,) = journal.records()

    assert back.refusals["roster"] == "panel covered"


def test_a_missing_journal_reads_as_empty(tmp_path):
    journal = Journal(tmp_path / "never-written.jsonl")

    assert list(journal.records()) == [] and len(journal) == 0


# ---------------------------------------------------------------- the round ----

def reading(*, roster=None, deck=None) -> FrameReading:
    return FrameReading(at=0.0, screen="table", roster=roster, deck_remaining=deck)


def test_a_new_roster_is_a_new_round():
    tracker = RoundTracker()

    assert tracker.observe(reading(roster=("gen1", "gen4", "id3", "myth"))) == 0
    assert tracker.observe(reading(roster=("gen1", "gen4", "id3", "myth"))) == 0
    assert tracker.observe(reading(roster=("gamers", "gen4", "gen5", "myth"))) == 1


def test_a_deck_counter_going_up_to_a_fresh_deck_is_a_new_round():
    """It only ever falls within a round, so a rise to a *full* deck can only be a fresh deal.

    Needed as well as the roster because the roster is unreadable while a payout covers the
    panel -- and a payout is exactly what tends to be on screen when a round ends.
    """
    tracker = RoundTracker()
    tracker.observe(reading(deck=40))
    tracker.observe(reading(deck=12))

    assert tracker.round_index == 0
    assert tracker.observe(reading(deck=DECK_BEFORE_DEAL)) == 1


def test_a_rise_that_lands_short_of_a_fresh_deck_is_a_misread():
    """The failure this exists for, seen live and expensive.

    A payout display partly covering the counter turned 65 into a confident 1. The rise back
    to 65 cleared `NEW_DEAL_RISE` easily and read as a new deal, which restarted the coin
    ledger at the opening 1000 in the middle of a round and cost tracking for the next 56
    frames -- a whole round unadvisable, from one misread digit. A deal leaves
    `DECK_BEFORE_DEAL` cards, so a rise landing below that is not one however large it is.
    """
    tracker = RoundTracker()
    tracker.observe(reading(deck=65))
    tracker.observe(reading(deck=1))            # the misread

    assert tracker.observe(reading(deck=65)) == 0
    assert tracker.round_index == 0


def test_a_small_rise_is_not_a_new_round():
    """A single misread digit can invent a rise, so the jump has to mean something."""
    tracker = RoundTracker()
    tracker.observe(reading(deck=40))

    assert tracker.observe(reading(deck=40 + NEW_DEAL_RISE)) == 0


def test_a_falling_deck_stays_in_the_round():
    tracker = RoundTracker()

    assert [tracker.observe(reading(deck=d)) for d in (71, 63, 46, 30, 0)] == [0] * 5


def test_frames_that_read_nothing_do_not_move_the_round():
    """Most mid-payout frames read neither roster nor counter, and they are common. Advancing
    on absence would split one round into dozens."""
    tracker = RoundTracker()
    tracker.observe(reading(roster=("gen1", "gen4", "id3", "myth"), deck=40))

    for _ in range(5):
        assert tracker.observe(reading()) == 0


def test_the_round_count_never_retreats():
    """A wrongly split round is recoverable by joining two logs; a wrongly joined one silently
    mixes two decks, and the deck is what the entire belief is inferred against."""
    tracker = RoundTracker()
    tracker.observe(reading(roster=("gen1", "gen4", "id3", "myth"), deck=30))
    tracker.observe(reading(roster=("gamers", "gen4", "gen5", "myth"), deck=71))
    high = tracker.round_index

    tracker.observe(reading(roster=("gen1", "gen4", "id3", "myth"), deck=10))

    assert tracker.round_index >= high


# ------------------------------------------------------------------ the meld ----
#
# The meld is drawn toward the seat that called, so `layout.MELD_BOXES` holds one box per
# caller and they overlap. What matters is not that a meld is found -- `check_vision.py`
# measures that against real captures -- but what happens when more than one box answers.

def _art(seed: int, height: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, size=(8, 6, 3)).astype(np.float32)
    ys = np.linspace(0, 7, height).astype(int)
    xs = np.linspace(0, 5, width).astype(int)
    return coarse[np.ix_(ys, xs)].astype(np.uint8)


def _paint_meld(pixels, area, seat, cards=3, seed=7, grow=None):
    """`cards` identical cards filling one seat's meld box, on felt.

    `grow` paints the meld in a direction other than the one the seat is believed to use, for
    the seats where that belief is unproven.
    """
    from pokajan.vision.geometry import FRAME_REFERENCES

    # Painted straight into the boxes the reader will cut back out, since those are the cards'
    # own edges rather than a region to search within.
    boxes = layout.meld_cards(seat, cards, grow=grow)
    for box in boxes:
        left, top, right, bottom = box.pixels(area)
        height, width = bottom - top, right - left
        pixels[top:bottom, left:right] = FRAME_REFERENCES["blue"]
        by, bx = int(height * 0.10), int(width * 0.12)
        pixels[top + by:bottom - by, left + bx:right - bx] = _art(
            seed, height - 2 * by, width - 2 * bx)
    left, top, right, bottom = boxes[0].pixels(area)
    # The catalogue entry has to go through the same preparation a real one does, or the
    # correlation is between vectors of different lengths.
    return prepare(pixels[top:bottom, left:right])


def _meld_reader(art: np.ndarray) -> TableReader:
    return TableReader(cards=None, digits=FixedDigits(None),
                       templates=TemplateSet({"gawr_gura": [art]}))


def test_a_meld_is_read_with_its_seat():
    pixels, area = felt()
    art = _paint_meld(pixels, area, "right")

    melds = _meld_reader(art).read_meld(pixels, area)

    assert any(m.seat == "right" and len(m.cards) == 3
               and {c.character for c in m.cards} == {"gawr_gura"} for m in melds)


def test_candidates_are_returned_rather_than_chosen_between():
    """A shorter box fits inside a longer meld, so a group call yields a subset candidate too.

    Throwing one away here would mean choosing on the pixels alone. `accumulate` chooses with
    the ledger's winning seat and the amount, which is strictly more evidence — and it is what
    threw out the real subset: three of a four-member group is not a complete group.
    """
    pixels, area = felt()
    art = _paint_meld(pixels, area, "right", cards=4)

    melds = _meld_reader(art).read_meld(pixels, area)
    lengths = {len(m.cards) for m in melds if m.seat == "right"}

    assert 4 in lengths and 3 in lengths


def test_a_roster_hint_narrows_the_field_when_the_frame_cannot_read_its_own():
    """The gap this closes is total, not occasional: measured over a live round the roster
    reads on 59% of frames and on **0% of the frames showing a meld**, because the payout
    panel that reveals a meld is the same panel that covers the group list. Every saved meld
    fixture is the same. So the read that most needs a narrowed field never got one.

    Asserted through a hint that *excludes* the painted holomem, because that is the direction
    that proves the restriction reached the matcher — a hint that includes it would pass
    whether it was applied or not.
    """
    pixels, area = felt()
    art = _paint_meld(pixels, area, "right")
    reader = _meld_reader(art)
    assert reader.read(pixels, area).roster is None, "felt must not read a roster"

    named = reader.read(pixels, area)
    narrowed = reader.read(pixels, area, roster_hint=("gen2",))

    assert any("gawr_gura" in {c.character for c in m.cards} for m in named.melds)
    assert not narrowed.melds


def test_what_the_frame_itself_shows_outranks_the_hint(monkeypatch):
    """Evidence beats memory, and the failure this prevents is the expensive kind.

    A hint is the *previous* frames' answer, so a hint that outranks the panel would keep
    naming last round's fifteen holomem through a deal — restricting the field to a set the
    table does not contain, which is the one input that turns a refusal into a confident wrong
    answer. `_roster` is stubbed because plain felt reads no roster at all, so the precedence
    is invisible on it: both orderings agree when one side is always None, and a mutant that
    reversed them passed until this test existed.
    """
    pixels, area = felt()
    art = _paint_meld(pixels, area, "right")
    reader = _meld_reader(art)
    monkeypatch.setattr(reader, "_roster", lambda *a, **k: ("table", ("myth",)))

    reading = reader.read(pixels, area, roster_hint=("gen2",))

    assert reading.roster == ("myth",)
    assert any("gawr_gura" in {c.character for c in m.cards} for m in reading.melds), \
        "gawr_gura is in myth and not in gen2, so the panel's roster must have won"


def test_a_roster_hint_never_becomes_the_frames_roster():
    """The journal records frames, not conclusions, so that a later and better accumulator can
    re-derive rounds over the same log rather than inherit this one's guesses. A hint written
    into `roster` would be this round's answer masquerading as this frame's evidence — and
    `RoundTracker` starts a new round when the roster changes, so it would also quietly make
    every mid-payout frame agree with whatever the tracker already believed."""
    pixels, area = felt()
    art = _paint_meld(pixels, area, "right")

    reading = _meld_reader(art).read(pixels, area, roster_hint=("myth",))

    assert reading.roster is None
    assert any("gawr_gura" in {c.character for c in m.cards} for m in reading.melds), \
        "the hint should still have been used for matching"


def test_an_extension_that_finds_nothing_says_so():
    """Silence here hid a wrong meld box for four rounds and it must not come back.

    A longer candidate is only tried because the three-card slice already read, so the seat
    certainly has a meld and the only open question is its length. "The fourth card is bare
    felt" is the answer to that, not the absence of one — and for the three seats that grow
    leftward the new card is index 0, which is exactly the case the old rule kept quiet.

    That is what a bottom-seat five-group looked like in the log: a plain triple, no refusal,
    nothing to notice. The box was cut from felt on the wrong side of the meld.
    """
    pixels, area = felt()
    art = _paint_meld(pixels, area, "right")          # three cards only
    refusals: dict[str, str] = {}

    melds = _meld_reader(art).read_meld(pixels, area, refusals=refusals)

    assert any(m.seat == "right" and len(m.cards) == 3 for m in melds)
    assert not any(len(m.cards) > 3 for m in melds)
    assert "meld_right4_0" in refusals, f"the four-card attempt said nothing: {refusals}"


# ------------------------------------------------------------ whose turn ----
LIT = (255, 232, 60)          # the bar the game lights for the active seat
UNLIT = (45, 111, 17)         # measured off a real capture; not the felt around it


def _paint_turn(pixels, area, active=None):
    """The four bars around the central oval, with at most one lit."""
    for seat, box in layout.TURN_INDICATORS.items():
        left, top, right, bottom = box.pixels(area)
        pixels[top:bottom, left:right] = LIT if seat == active else UNLIT


def test_each_seats_bar_is_on_the_edge_of_the_table_nearest_that_seat():
    """Absolute positions, because the painted tests below cannot catch this.

    They paint through `TURN_INDICATORS` and read back through it, so swapping two seats' boxes
    is invisible to them — self-consistent and wrong, which is the same trap the meld anchors
    fell into. These are the measured coordinates: the lit bar sat at x 0.445-0.548, y
    0.278-0.307 on a capture where `top` was active, and at x 0.278-0.305, y 0.348-0.482 on one
    where `left` was.
    """
    boxes = layout.TURN_INDICATORS
    assert set(boxes) == set(layout.SEAT_ORDER)
    mid_x = sum(box.left + box.right for box in boxes.values()) / (2 * len(boxes))
    mid_y = sum(box.top + box.bottom for box in boxes.values()) / (2 * len(boxes))

    def centre(seat):
        box = boxes[seat]
        return (box.left + box.right) / 2, (box.top + box.bottom) / 2

    assert centre("top")[1] < mid_y and centre("bottom")[1] > mid_y
    assert centre("left")[0] < mid_x and centre("right")[0] > mid_x
    # And the two that were actually measured, to three decimals.
    assert (round(boxes["top"].left, 3), round(boxes["top"].top, 3)) == (0.445, 0.278)
    assert (round(boxes["left"].left, 3), round(boxes["left"].top, 3)) == (0.278, 0.348)


@pytest.mark.parametrize("seat", layout.SEAT_ORDER)
def test_the_lit_bar_names_the_seat_whose_turn_it_is(seat):
    """Four bars around the central oval, one per seat on the edge nearest it. Measured on two
    captures with the active seat known: the lit bar reads a yellow fraction of 0.33-0.38 and
    the other three read exactly 0.00."""
    pixels, area = felt()
    _paint_turn(pixels, area, active=seat)

    assert reader().read_turn(pixels, area) == seat


def test_no_bar_lit_is_the_ordinary_answer_and_not_an_error():
    """A payout panel covers the bars, and that is when the reader is looking hardest. Eleven of
    the eighteen captures on disk show no lit bar, all of them mid-animation."""
    pixels, area = felt()
    _paint_turn(pixels, area, active=None)
    refusals: dict[str, str] = {}

    assert reader().read_turn(pixels, area, refusals) is None
    assert "no turn indicator is lit" in refusals["current_seat"]


def test_two_bars_lit_is_refused_rather_than_broken_between():
    """The game has one active seat, so two lit is evidence the reading is wrong -- and picking
    the brighter would turn a broken read into a confident wrong player."""
    pixels, area = felt()
    _paint_turn(pixels, area, active="top")
    left, top, right, bottom = layout.TURN_INDICATORS["left"].pixels(area)
    pixels[top:bottom, left:right] = LIT
    refusals: dict[str, str] = {}

    assert reader().read_turn(pixels, area, refusals) is None
    assert "2 turn indicators are lit" in refusals["current_seat"]


def test_the_unlit_bar_is_not_mistaken_for_the_felt_behind_it():
    """The bottom seat has never been caught lit -- neither capture is the player's own turn --
    so its box is placed on the *unlit* bar's colour instead. That is a real distinction and not
    a fudge: unlit reads (45,111,17) against felt at (35,123,0), whose blue channel is 0. If the
    box were on felt this would still pass, so it also checks the felt is not read as lit."""
    pixels, area = felt()          # no bars painted at all
    assert reader().read_turn(pixels, area) is None


def test_bare_felt_shows_no_meld():
    pixels, area = felt()
    art = _paint_meld(pixels.copy(), area, "right")
    assert _meld_reader(art).read_meld(pixels, area) == ()


def test_every_seat_has_a_meld_position():
    assert set(layout.MELD_ANCHORS) == set(layout.SEAT_ORDER)


@pytest.mark.parametrize("seat", layout.SEAT_ORDER)
def test_a_longer_meld_grows_from_the_edge_that_stays_put(seat):
    """Measured per seat, because it is **not** the same for all four and assuming it was cost
    four rounds of five-card calls.

    The right seat's four-card meld ran x 0.465-0.676 against its three-card melds at
    0.518-0.676 — same right edge, and 0.676 - 4 x 0.0527 = 0.465. The bottom seat goes the
    other way: on a +480 five-group and a +300 four-group nothing frame-coloured existed left
    of 0.365, while content ran on past the old anchor to 0.636.
    """
    three, five = layout.meld_box(seat, 3), layout.meld_box(seat, 5)
    grows_right = seat in layout.MELD_GROWS_RIGHT

    assert (three.left == five.left) is grows_right
    assert (three.right == five.right) is not grows_right
    assert round(five.right - five.left, 4) == round(5 * layout.MELD_CARD, 4)


def test_the_bottom_meld_occupies_the_span_a_five_group_was_measured_in():
    """Absolute numbers, because the test above cannot catch this one.

    That test asserts `meld_cards` agrees with `MELD_GROWS_RIGHT`, and a mutation that flips
    the seat flips both together — it passed against exactly the regression this file exists
    to prevent. Self-consistency is not evidence; these are the pixels.

    From the survey on the frames where the bottom seat took +480 for a five-group and +300
    for a four-group: the leftmost frame-coloured edge was 0.365 on both, nothing existed
    below it, and content ran on past the old 0.523 anchor to 0.636.
    """
    five = layout.meld_box("bottom", 5)

    assert round(five.left, 3) == 0.365, "nothing frame-coloured was ever seen left of this"
    assert 0.62 <= five.right <= 0.64, "content ran to 0.636 on the five-group frame"


@pytest.mark.parametrize("seat, edge", [("right", 0.676), ("top", 0.630)])
def test_the_confirmed_leftward_seats_keep_their_measured_right_edge(seat, edge):
    """Only the two seats where a four-card group call was actually read and matched its
    amount. Their predicted edges — 0.518/0.571/0.623/0.676 and 0.472/0.525/0.577/0.630 —
    match the survey to three decimals.

    `left` is deliberately not in this list. It was, briefly, on nothing but the pattern these
    two set, which is the same reasoning that got the bottom seat wrong. Fourteen left melds
    have been read and every one was three cards, the length that cannot tell the directions
    apart. See the unsettled-seat tests below.
    """
    for cards in layout.MELD_SIZES:
        assert round(layout.meld_box(seat, cards).right, 3) == edge


def test_a_seat_whose_direction_is_unproven_is_cut_both_ways(monkeypatch):
    """Every seat is settled now, so this drives the machinery through a seat forced unsettled.

    Kept because it is how a seat gets read *at all* while its direction is in doubt, and the
    doubt is not hypothetical: `left` sat here until a call finally proved it, having shown
    fourteen melds that were all three cards — the length at which the two readings are the
    same three boxes and cannot disagree.

    Rather than pick, cut both and let `accumulate` choose on the payout amount, the same
    reason candidates are not chosen between by *length* either.
    """
    monkeypatch.setattr(layout, "MELD_UNSETTLED", frozenset({"right"}))
    assert layout.meld_growths("right") == ("left", "right")
    assert layout.meld_growths("top") == ("left",), "settled seats take one direction only"

    # Rounded, because the two arithmetic paths differ in the last float bit -- 0.3756 against
    # 0.37559999999999993 -- which is a hundredth of a pixel and vanishes in `Box.pixels`.
    def edges(cards, grow):
        return [(round(b.left, 9), round(b.right, 9))
                for b in layout.meld_cards("right", cards, grow=grow)]

    three = layout.MELD_SIZES[0]
    assert edges(three, "left") == edges(three, "right"), \
        "the two readings agree exactly at three cards -- that is the whole trap"
    assert edges(4, "left") != edges(4, "right")


def test_an_unsettled_seat_reads_a_meld_grown_the_unexpected_way(monkeypatch):
    """The point of cutting both: the meld must read whichever way it actually grew.

    Painted in the direction the seat is *not* believed to use, so this fails if `read_meld`
    only ever cuts what the seat is assumed to do.
    """
    monkeypatch.setattr(layout, "MELD_UNSETTLED", frozenset({"right"}))
    pixels, area = felt()
    art = _paint_meld(pixels, area, "right", cards=4, grow="right")

    melds = _meld_reader(art).read_meld(pixels, area)

    assert any(m.seat == "right" and len(m.cards) == 4 for m in melds), \
        f"read {[(m.seat, len(m.cards)) for m in melds]}"


@pytest.mark.parametrize("seat", layout.SEAT_ORDER)
def test_a_longer_meld_contains_its_own_three_card_slice(seat):
    """What makes the gate sound, and what makes the direction readable off a log.

    The three-card slice is cut from the same fixed edge, so a longer meld always contains it —
    as its **first** three cards where the meld grows rightward, as its **last** three where it
    grows leftward. That is why a seat showing no three-card meld shows none at all, and it is
    also how the direction was settled with no pixels involved: across every meld ever logged
    alongside its own slice, 2/0 rightward for `left`, 3/0 for `bottom`, 0/19 for `top` and
    0/16 for `right`, with nothing contradicting.
    """
    def edges(boxes):
        return [(round(b.left, 9), round(b.right, 9)) for b in boxes]

    three = edges(layout.meld_cards(seat, layout.MELD_SIZES[0]))
    five = edges(layout.meld_cards(seat, 5))

    if seat in layout.MELD_GROWS_RIGHT:
        assert five[:3] == three
    else:
        assert five[-3:] == three


def test_the_left_seats_three_card_span_survived_being_reclassified():
    """`left` moved from leftward to rightward growth, which moves its anchor from the right
    edge of the three-card span to the left edge. Fourteen left triples had already been read
    correctly against the old value, so the span itself must not have shifted by a pixel — if
    it had, the reclassification would have quietly broken the one case that was working."""
    assert tuple(round(v, 4) for v in layout.meld_span("left")) == (0.3229, 0.4810)


@pytest.mark.parametrize("seat", layout.SEAT_ORDER)
@pytest.mark.parametrize("cards", layout.MELD_SIZES)
def test_the_cards_of_a_meld_start_at_the_anchor_and_tile_away_from_it(seat, cards):
    """Pinned against the anchor itself rather than against the other geometry helper.

    The synthetic tests above paint through `meld_cards` and read back through it, so they
    hold whether the row is anchored on its right edge or its left — self-consistent and
    blind. Three mutants proved it: reversing the direction, shifting the whole row, and
    finally the real thing, a seat whose meld genuinely grows the other way. The anchor is the
    measured fact, so that is what to assert on.
    """
    boxes = layout.meld_cards(seat, cards)
    anchor = layout.MELD_ANCHORS[seat][0]

    assert len(boxes) == cards
    if seat in layout.MELD_GROWS_RIGHT:
        assert boxes[0].left == anchor              # first card starts at the anchor, always
        assert boxes[-1].right == anchor + cards * layout.MELD_CARD
    else:
        assert boxes[-1].right == anchor            # last card ends at the anchor, always
        assert boxes[0].left == anchor - cards * layout.MELD_CARD
    for near, far in zip(boxes, boxes[1:]):         # tiled left to right, no gaps
        assert near.right == far.left
    # And they tile exactly the box the other helper describes.
    whole = layout.meld_box(seat, cards)
    assert (boxes[0].left, boxes[-1].right) == (whole.left, whole.right)


def test_a_three_card_meld_is_the_same_box_whichever_way_it_grows():
    """The coincidence that hid a wrong box for four rounds, pinned so it stays understood.

    The bottom anchor was calibrated on a triple. A triple is three cards from a fixed edge,
    and the *other* reading is three cards from the edge three cards away — the same three
    boxes. So every bottom triple read perfectly while every bottom four- and five-card call
    was cut from bare felt, and nothing in the log distinguished the two until a five-group
    was surveyed. Two seats agreeing is not a rule; a length that cannot disagree is not
    evidence.
    """
    edge, top, bottom = layout.MELD_ANCHORS["bottom"]
    mirrored = {"bottom": (edge + 3 * layout.MELD_CARD, top, bottom)}

    with_growth = layout.meld_cards("bottom", 3)
    against = tuple(reversed([
        layout.Box(mirrored["bottom"][0] - k * layout.MELD_CARD, top,
                   mirrored["bottom"][0] - (k - 1) * layout.MELD_CARD, bottom)
        for k in range(3, 0, -1)]))

    assert [(round(b.left, 6), round(b.right, 6)) for b in with_growth] \
        == sorted((round(b.left, 6), round(b.right, 6)) for b in against)


def test_the_bottom_meld_box_stops_short_of_the_payout_panel():
    """Measured, and the one edge with no slack.

    A bottom-seat meld sits directly above the winner's payout panel, which is near-white. Let
    the box reach past about 0.671 and that panel joins the frame-colour ring, so the colours
    stop reading -- while the holomem still reads, because a triple is three of one holomem and
    identification cannot notice a slice that has drifted. Widening this "to be safe" is the
    natural mistake and it is the wrong direction.
    """
    assert layout.meld_box("bottom", 3).bottom <= 0.671
