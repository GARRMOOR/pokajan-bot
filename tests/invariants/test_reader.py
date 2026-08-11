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
from pokajan.vision.templates import TemplateSet

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


def test_a_deck_counter_going_up_is_a_new_round():
    """It only ever falls within a round, so a rise can only be a fresh deal.

    Needed as well as the roster because the roster is unreadable while a payout covers the
    panel -- and a payout is exactly what tends to be on screen when a round ends.
    """
    tracker = RoundTracker()
    tracker.observe(reading(deck=40))
    tracker.observe(reading(deck=12))

    assert tracker.round_index == 0
    assert tracker.observe(reading(deck=12 + NEW_DEAL_RISE + 1)) == 1


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
