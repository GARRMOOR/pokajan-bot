"""Turning one settled frame into everything currently readable from it.

This is the precursor to the full `TableReader`, and the distinction is the whole design.
A `FrameReading` is what a **single frame** can support, and that is deliberately less than
a `PublicState`: `table` and `scored` cannot come from one frame at all. Buried discards are
unreadable once a pile stacks, and a scored meld leaves the table entirely, so both have to
be accumulated turn by turn from the deal. There is no clever frame to catch up from.

So this module answers "what is on screen", and something above it answers "what has
happened". Keeping them apart is what stops the second quietly pretending to know the first.

**Everything is optional and every absence is explained.** A field is either read or it is in
`refusals` with the reason -- never silently defaulted. The reason matters more than it
sounds: an advisor with a wrong coin total gives confident advice about the wrong game, so a
caller has to be able to tell "seat two has 780" from "seat two could not be read" without
inspecting sentinel values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import layout
from .digits import DigitReader
from .geometry import classify_colour, find_row, frame_colour, frame_spans
from .group_labels import LabelReader
from .roster_panel import GroupBook, PanelError, read_roster
from .templates import TemplateSet

CARDS_DIR = Path(__file__).resolve().parents[2] / "data" / "cards"

# Sanity bound on the deck counter: the deck holds 100, so anything above that is a misread.
#
# This was 72 -- 100 less the 28-card opening deal -- and that was wrong. A live capture read
# 100 and had it refused, because **the counter is on screen before the deal happens**, showing
# the full deck. Guessing the bound from arithmetic about a state the reader had never actually
# watched cost a legitimate reading, which is the milder version of the usual failure here but
# the same mistake.
MAX_DECK_REMAINING = 100
# Above this the deal has not happened yet, so there is no hand to read and no coins to check.
# Useful rather than merely tolerated: it is the cleanest "a round is starting" signal there
# is, and unlike the roster it stays readable while a payout covers the panel.
DECK_BEFORE_DEAL = 72


@dataclass(frozen=True)
class Card:
    """One card as read: who is on it, what colour its frame is."""

    character: str | None
    colour: str | None
    score: float = 0.0
    margin: float = 0.0

    @property
    def known(self) -> bool:
        return self.character is not None and self.colour is not None

    def __str__(self) -> str:
        return f"{self.character or '?'}:{self.colour or '?'}"


@dataclass(frozen=True)
class Meld:
    """A face-up scoring meld, and which seat's position it was found in.

    A candidate, not a conclusion. `accumulate` decides which of a frame's candidates is real,
    using the winning seat and the amount from the coin ledger -- both of which this layer has
    no access to and should not guess at.
    """

    seat: str
    cards: tuple[Card, ...]

    def __str__(self) -> str:
        return f"{self.seat}: {' '.join(str(card) for card in self.cards)}"


@dataclass(frozen=True)
class FrameReading:
    """What one settled frame supports. Absent fields are explained in `refusals`."""

    at: float
    screen: str                              # "table" or "reveal"
    deck_remaining: int | None = None
    coins: dict[str, int] = field(default_factory=dict)
    hand: tuple[Card, ...] = ()
    roster: tuple[str, ...] | None = None    # group ids, top to bottom
    bonus: str | None = None
    # Who the bonus card looked most like, whether or not that was confident enough to accept.
    # A refusal is not the same as no information: the bonus does not change within a round, so
    # `accumulate.RoundBelief` votes across frames on this. Never an answer on its own.
    bonus_best: str | None = None
    # What is *visible* in each seat's discard field, which is not what that seat has thrown:
    # the near end of a pile buries its oldest cards. Only the two axis-aligned seats appear
    # here at all. `accumulate.DiscardFloor` is what turns a stream of these into a claim.
    discards: dict[str, tuple[Card, ...]] = field(default_factory=dict)
    # Face-up meld candidates, on the frames that show a payout. Empty everywhere else, and a
    # candidate is dropped rather than kept partial when any card of it refused. More than one
    # may be present: `accumulate` picks with the ledger's winning seat and amount.
    melds: tuple[Meld, ...] = ()
    # Whose turn it is, from the lit bar on the central oval's edge. None on most frames rather
    # than rarely: a payout covers the bars, and so does anything else drawn over the table.
    current_seat: str | None = None
    refusals: dict[str, str] = field(default_factory=dict)

    @property
    def coins_total(self) -> int | None:
        """The four seats' coins summed, or None unless all four were read.

        A partial sum is worse than no sum: it would fail the 4000-plus-minted check and
        look like a rules problem rather than a missing read.
        """
        if len(self.coins) != len(layout.SEAT_ORDER):
            return None
        return sum(self.coins.values())


def deck_key(digits: DigitReader):
    """A cheap turn clock for `capture.TurnGate`: the deck counter's value, or None.

    1.7 ms against 153 ms for a full read, and it is the game's own notion of a turn -- the
    counter falls by one on every draw, including the refill after a call. None when the
    counter cannot be read, which is itself a distinct key: it means something is covering the
    pile, and that something is usually the payout display.
    """
    def key(grab) -> object:
        return digits.read(grab.crop(layout.DECK_COUNTER), max_digits=3).value

    return key


class TableReader:
    """Reads what a frame shows. Holds the catalogues, not the game state.

    Stateless on purpose: the thing that accumulates `table` across turns has to own its own
    memory and be able to say when it has lost track, and mixing that into the per-frame read
    is how a reader ends up unable to tell a fresh observation from a remembered one.
    """

    def __init__(
        self,
        *,
        cards: str | Path | None = CARDS_DIR,
        book: GroupBook | None = None,
        digits: DigitReader | None = None,
        labels: LabelReader | None = None,
        templates: TemplateSet | None = None,
    ) -> None:
        self.book = book or GroupBook.load()
        self.digits = digits or DigitReader.load()
        self.labels = labels or LabelReader.load()
        if templates is not None:
            self.templates = templates
        else:
            directory = Path(cards) if cards else None
            self.templates = (TemplateSet.load(directory)
                              if directory and directory.is_dir() else TemplateSet({}))

    # ------------------------------------------------------------ the whole ---
    def read(self, frame: np.ndarray, area: layout.PlayArea, *,
             at: float = 0.0,
             roster_hint: tuple[str, ...] | None = None) -> FrameReading:
        """Everything readable, with a reason recorded for everything that is not.

        `roster_hint` is this round's group ids from somewhere other than this frame, and it
        exists because of a gap that is total rather than occasional: the roster reads on 59%
        of frames in a round but on **0% of the frames that show a meld** -- in a live round
        and on every saved meld fixture alike. The payout panel that reveals a meld is the
        same panel that covers the group list. So the one read that most needs a narrowed
        field is the only one that never gets it, and cards are ranked against all 89 holomem
        at the exact moment `scored` is observable.

        Only consulted when this frame's own roster refuses, and it never reaches
        `FrameReading.roster` -- that field stays what *this frame* showed, because the
        journal is a record of frames and not of conclusions, and a later accumulator has to
        be able to re-derive rounds over it rather than inherit this one's guesses.

        The caller owns the risk. A hint from the wrong round restricts the field to fifteen
        holomem that are not on the table, and a wrong field is the one input that turns a
        refusal into a confident wrong answer. `accumulate.RoundBelief` is per-round and
        settles its roster by agreement across dozens of frames, which is why it is the thing
        that supplies this.

        The screen is worked out here rather than passed in, and only two things turn on it.
        The reveal screen has no deck counter, no coin displays and no hand, so those are not
        attempted there -- a refusal list full of "nothing written here" for regions that
        legitimately do not exist is the same cry-wolf failure as a filename report that is
        always wrong: it teaches the reader of the log to skip the refusals.

        A frame whose roster refuses under **both** layouts is still treated as a table.
        That is the mid-payout case, and abandoning it would be the worst possible choice:
        those are the only frames on which a payout is observable at all, and a call's meld
        is the one moment `scored` can be seen before those cards leave the table for good.
        """
        refusals: dict[str, str] = {}
        screen, roster = self._roster(frame, area, refusals)
        if screen == "reveal":
            # The bonus holomem is a large card here too, but in its own place and often
            # still face-down; it needs a REVEAL_BONUS box that does not exist yet.
            return FrameReading(at=at, screen=screen, roster=roster, refusals=refusals)

        # Every card on the table belongs to one of this round's four groups, so the roster
        # narrows the field the matcher ranks against from the whole catalogue to about fifteen.
        # Measured: same answers, margins equal or better everywhere, and the one genuinely bad
        # crop still refused. When this frame's roster could not be read the round's own answer
        # stands in if the caller has one; failing that the field stays open rather than guessed.
        among = self.members(roster or roster_hint)
        bonus, bonus_best = self._bonus(frame, area, refusals, among)
        return FrameReading(
            at=at,
            screen=screen,
            deck_remaining=self._deck(frame, area, refusals),
            coins=self._coins(frame, area, refusals),
            hand=self._hand(frame, area, refusals, among),
            roster=roster,
            bonus=bonus,
            bonus_best=bonus_best,
            discards=self._discards(frame, area, among),
            melds=self.read_meld(frame, area, among=among, refusals=refusals),
            current_seat=self.read_turn(frame, area, refusals),
            refusals=refusals,
        )

    def read_turn(self, frame: np.ndarray, area: layout.PlayArea,
                  refusals: dict[str, str] | None = None) -> str | None:
        """Whose turn it is, from the four bars around the central oval.

        One bar per seat on the edge nearest it, the active one lit yellow and the rest a dull
        green -- see `layout.TURN_INDICATORS` for the measurements. Cheap enough to run on every
        frame: four small crops and a threshold, no matching.

        Two bars lit is not a tie to break, it is evidence the reading is wrong -- the game has
        one active seat -- so it refuses. None is the ordinary answer rather than the alarming
        one: a payout panel covers the bars, which is exactly when the reader is looking hardest.
        """
        lit: dict[str, float] = {}
        for seat, box in layout.TURN_INDICATORS.items():
            crop = area.crop(frame, box)
            if crop.size == 0:
                continue
            pixels = crop.astype(np.int16)
            red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
            share = float(((red > 190) & (green > 190) & (abs(red - green) < 45)
                           & (blue < 150)).mean())
            if share >= layout.TURN_LIT:
                lit[seat] = share
        if len(lit) == 1:
            return next(iter(lit))
        if refusals is not None:
            refusals["current_seat"] = (
                f"{len(lit)} turn indicators are lit ({', '.join(sorted(lit))})"
                if lit else "no turn indicator is lit -- something is drawn over the table"
            )
        return None

    def members(self, roster: tuple[str, ...] | None) -> tuple[str, ...] | None:
        """The holomem this round can deal, or None when the roster is unread."""
        if roster is None:
            return None
        by_id = {group.id: group for group in self.book.groups}
        found = [m for gid in roster if gid in by_id for m in by_id[gid].members]
        return tuple(found) or None

    # ----------------------------------------------------------- the pieces ---
    def _deck(self, frame, area, refusals) -> int | None:
        number = self.digits.read(area.crop(frame, layout.DECK_COUNTER), max_digits=3)
        if not number.confident:
            refusals["deck_remaining"] = number.reason
            return None
        if number.value > MAX_DECK_REMAINING:
            # The deck holds 100, so this cannot be a real count. Refusing a plausible number
            # is the whole job: a wrong deck count moves every belief in the game.
            refusals["deck_remaining"] = (
                f"read {number.value}, above the {MAX_DECK_REMAINING} cards the deck holds "
                f"-- misread rather than a surprising game"
            )
            return None
        return number.value

    def _coins(self, frame, area, refusals) -> dict[str, int]:
        found: dict[str, int] = {}
        for seat in layout.SEAT_ORDER:
            number = self.digits.read(area.crop(frame, layout.COINS[seat]), max_digits=5)
            if number.confident:
                found[seat] = number.value
            else:
                refusals[f"coins_{seat}"] = number.reason
        return found

    def read_discards(self, frame: np.ndarray, area: layout.PlayArea, seat: str, *,
                      among: tuple[str, ...] | None = None) -> tuple[Card, ...]:
        """The cards visible in one seat's discard field, newest last.

        Only the two axis-aligned seats. The seats either side lay their discards out as
        sheared diagonal staircases, so `find_row` does not describe them and pointing it at
        them returns plausible nonsense rather than nothing -- see this package's `__init__`.

        Note what this is *not*: `table`. Older discards slide under the near end of the pile as
        it grows and become unreadable, so what is visible here is the recent tail, not the
        history. Accumulating that history is the caller's job.
        """
        if seat not in layout.DISCARD_ASPECT:
            raise KeyError(
                f"{seat!r} lays its discards out as a diagonal staircase, not a row -- "
                f"find_row does not describe it"
            )
        row = find_row(area.crop(frame, layout.DISCARDS[seat]),
                       aspect=layout.DISCARD_ASPECT[seat],
                       rotate=180 if seat == "top" else 0)
        if row is None:
            return ()
        found: list[Card] = []
        for card in row.cards:
            match = self.templates.identify(card, among=among)
            colour, _ = classify_colour(frame_colour(card))
            found.append(Card(match.character, colour, match.score, match.margin))
        return tuple(found)

    def read_meld(self, frame: np.ndarray, area: layout.PlayArea, *,
                  among: tuple[str, ...] | None = None,
                  refusals: dict[str, str] | None = None) -> tuple["Meld", ...]:
        """Every face-up meld this frame could be showing, one per (seat, length) that fits.

        The one moment `scored` is observable. The coin displays give the amount, and the
        amount pins the shape to one or two candidates and no further -- a monochrome triple
        and a monochrome four-group both pay 840. These are the cards themselves.

        **Every card must be confidently identified or nothing is returned.** A partial meld
        is worse than none: `scored` is a count vector, so a half-read meld would either drop
        a card that has left the game or, worse, be merged with the payout's shape candidates
        and appear to confirm one of them.

        A meld grows from a fixed edge, so a candidate is a seat and a length: four seats by
        three legal lengths. Which edge is per-seat and not guessable -- three seats grow
        leftward and the player's own grows rightward, see `layout.MELD_GROWS_RIGHT`. Each card
        is **cut straight out of known geometry** rather than segmented -- see the same note. A
        misplaced cut then shows up as a card that will not identify, which is already handled,
        instead of as a count that quietly comes out one short.

        **Deliberately does not choose between them**, and two of the overlaps are real rather
        than theoretical. A shorter box fits inside a longer meld, so the right seat's four-card
        group also fills its own three-card box with the rightmost three of those cards -- that
        happened, and `accumulate.meld_shape` is what threw the subset out, since three of a
        four-member group is not a complete group. Choosing here would mean choosing on the
        pixels alone; `accumulate` chooses with the ledger's winning seat and the amount, which
        is strictly more evidence.

        Returning everything is safe because the guards are downstream and strict: every card
        must be identified, the shape must be a real one, and it must pay what the coins say was
        paid. Pointed at an ordinary table frame these regions hold the deck's decoy list and
        felt, and measured on four such captures not one of the twelve boxes yields a candidate.
        """
        found: list[Meld] = []
        for seat in layout.MELD_ANCHORS:
            for size in layout.MELD_SIZES:
                # The shortest slice gates the longer ones. A meld grows from a fixed edge, so a
                # four- or five-card one *contains* the three-card slice -- as its last three
                # cards where the meld grows leftward, as its first three where it grows right.
                # Either way, if those will not read then neither will the longer candidates,
                # which include them. So a seat showing no three-card meld shows none at all.
                #
                # Worth the care because this is not free: twelve candidates of up to five
                # identifies each took a full read from 153 ms to 360 ms, which halved the
                # sampling rate and so cost meld sightings -- the opposite of the point. The
                # common case is now four slices, not twelve.
                shortest = size == layout.MELD_SIZES[0]
                if not shortest and not any(m.seat == seat for m in found):
                    continue
                # A seat whose growth direction has never been checked against a long call gets
                # both directions cut, and `accumulate` picks with the payout amount. The
                # shortest size takes only one, because there the two agree *exactly*: the
                # candidate fixed edges are three cards apart, so a three-card meld is the same
                # three boxes either way. That coincidence is what hid the bottom seat's
                # direction for four rounds; here it just means there is nothing to re-cut.
                growths = layout.meld_growths(seat)
                for grow in (growths[:1] if shortest else growths):
                    meld = self._meld_candidate(frame, area, seat, size, grow,
                                                among=among, refusals=refusals)
                    if meld is not None:
                        found.append(meld)
        return tuple(found)

    def _meld_candidate(self, frame: np.ndarray, area: layout.PlayArea, seat: str, size: int,
                        grow: str, *, among: tuple[str, ...] | None,
                        refusals: dict[str, str] | None) -> Meld | None:
        """One seat-by-length-by-direction slice, or None with a reason written down."""
        cards: list[Card] = []
        # Named so two directions cannot overwrite each other's reason, and only when there
        # are two -- otherwise every key in every old log would change meaning.
        tag = f"{seat}{size}" + (grow[0] if seat in layout.MELD_UNSETTLED else "")
        for index, box in enumerate(layout.meld_cards(seat, size, grow=grow)):
            crop = area.crop(frame, box)
            match = self.templates.identify(crop, among=among)
            colour, _ = classify_colour(frame_colour(crop))
            if match.character is None or colour is None:
                # Why a candidate died, when the answer is worth having. Two cases are: a slice
                # that holds a card nobody can name, and a candidate that had already read a
                # card and then broke. Both mean something was there.
                #
                # A slice of plain felt is neither, and that is the common case -- most of the
                # twelve are empty on any frame -- so it stays silent. Keying that on `index`
                # was the first attempt and it was too narrow: a meld whose *rightmost* card
                # fails is reported by nothing at all, which is exactly what happened to a
                # four-card and a five-card call in one round. `colour` is the better test,
                # since a classified frame colour means a card is present whether or not it
                # can be identified.
                #
                # An **extension** is always reported, whatever it found. A longer candidate is
                # only tried at all because the three-card slice already read, so the seat
                # certainly has a meld and the only question is how long -- "the fourth card is
                # bare felt" is the answer, not an absence of one. Staying quiet there hid a
                # wrong box for four rounds: the bottom seat's meld grows the other way, so
                # every extension was cut from felt, and a silent break looks exactly like an
                # ordinary triple.
                extension = size != layout.MELD_SIZES[0]
                if refusals is not None and (index or colour is not None or extension):
                    refusals[f"meld_{tag}_{index}"] = match.reason or (
                        "frame colour is not blue, orange or pink")
                return None
            cards.append(Card(match.character, colour, match.score, match.margin))
        return Meld(seat, tuple(cards))

    def survey_cards(self, frame: np.ndarray, area: layout.PlayArea) -> tuple[str, ...]:
        """Where frame-coloured things sit in the middle of the table, as text.

        Temporary scaffolding with one job: find out where the payout meld is drawn for each
        caller. `layout.PAYOUT_MELD` locates a **right-seat** meld -- both captures that pinned
        it turn out to have been called by the same seat -- and nothing is known about the other
        three, nor about how a four- or five-card row extends.

        Answering that needs examples, and the cheap source of examples is ordinary play rather
        than a screenshotting session: the coin ledger already knows exactly which frames are
        mid-payout and which seat won, so a line of geometry logged on those frames is a
        labelled example. Spans in play-area fractions, which is the form the answer is wanted
        in anyway.

        Delete this, `layout.TABLE_INTERIOR` and `geometry.frame_spans` once the meld boxes are
        known.
        """
        spans = frame_spans(area.crop(frame, layout.TABLE_INTERIOR))
        box = layout.TABLE_INTERIOR
        wide, tall = box.right - box.left, box.bottom - box.top
        return tuple(
            f"x{box.left + x0 * wide:.3f}-{box.left + x1 * wide:.3f} "
            f"y{box.top + y0 * tall:.3f}-{box.top + y1 * tall:.3f}"
            for x0, x1, y0, y1 in spans
        )

    def _discards(self, frame, area, among=None) -> dict[str, tuple[Card, ...]]:
        """Each readable seat's visible discard field.

        No refusal is recorded for an empty field. A seat that has not discarded yet has
        nothing there, and that is by far the commonest reason `find_row` finds no row -- so a
        refusal here would fire on every frame of the opening and train a reader of the log to
        skip the refusal list, which is the one thing it exists to be read for. The seats that
        genuinely cannot be read are named once, in `accumulate.DiscardFloor.unread`, rather
        than once per frame.
        """
        found: dict[str, tuple[Card, ...]] = {}
        for seat in layout.DISCARD_ASPECT:
            cards = self.read_discards(frame, area, seat, among=among)
            if cards:
                found[seat] = cards
        return found

    def _hand(self, frame, area, refusals, among=None) -> tuple[Card, ...]:
        # Your own hand is closest to the camera, so this is the one region CARD_ASPECT was
        # measured for and the only one that needs no correction.
        row = find_row(area.crop(frame, layout.HAND))
        if row is None:
            refusals["hand"] = "no row of cards in the hand region"
            return ()
        cards: list[Card] = []
        for index, card in enumerate(row.cards):
            match = self.templates.identify(card, among=among)
            colour, _ = classify_colour(frame_colour(card))
            if match.character is None:
                refusals[f"hand_{index}"] = match.reason
            if colour is None:
                refusals[f"hand_{index}_colour"] = "frame colour is not blue, orange or pink"
            cards.append(Card(match.character, colour, match.score, match.margin))
        return tuple(cards)

    def _bonus(self, frame, area, refusals, among=None) -> tuple[str | None, str | None]:
        row = find_row(area.crop(frame, layout.BONUS_CARD), expected=1)
        if row is None or not row.cards:
            refusals["bonus"] = "no card in the bonus region"
            return None, None
        match = self.templates.identify(row.cards[0], among=among)
        if match.character is None:
            refusals["bonus"] = match.reason
        return match.character, match.best

    # -------------------------------------------------------- which screen ---
    def _roster(self, frame, area,
                refusals) -> tuple[str, tuple[str, ...] | None]:
        """Which screen this is, and the roster if it could be read.

        Decided by *which layout produces a roster*, because that is the most heavily
        cross-checked thing either screen offers: four confident badges and four member
        counts that independently agree with them. Neither layout yields a single confident
        badge on the other's screen, measured across every capture.

        Deliberately not a chain that keeps whatever does not outright fail. The table panel
        box counts [5, 5, 5, 5] on the reveal screen -- four perfectly legal group sizes -- so
        a check resting on the counts alone would call that a table. It is the badges refusing
        that separates them.

        Neither succeeding means "table, panel covered", not "unknown screen": the reveal
        screen is a still presentation with nothing to occlude it, while the table spends much
        of a round under a payout display.
        """
        reasons: list[str] = []
        for screen, panel_box, labels_box in (
            ("table", layout.GROUP_PANEL, layout.GROUP_LABELS),
            ("reveal", layout.REVEAL_PANEL, layout.REVEAL_LABELS),
        ):
            try:
                roster = read_roster(area.crop(frame, panel_box),
                                     area.crop(frame, labels_box),
                                     book=self.book, reader=self.labels)
            except PanelError as error:
                reasons.append(f"{screen}: {error}")
                continue
            return screen, tuple(g.id for g in roster.groups)

        refusals["roster"] = " | ".join(reasons)
        return "table", None
