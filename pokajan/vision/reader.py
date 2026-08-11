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
from .geometry import classify_colour, find_row, frame_colour
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
class FrameReading:
    """What one settled frame supports. Absent fields are explained in `refusals`."""

    at: float
    screen: str                              # "table" or "reveal"
    deck_remaining: int | None = None
    coins: dict[str, int] = field(default_factory=dict)
    hand: tuple[Card, ...] = ()
    roster: tuple[str, ...] | None = None    # group ids, top to bottom
    bonus: str | None = None
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
             at: float = 0.0) -> FrameReading:
        """Everything readable, with a reason recorded for everything that is not.

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

        return FrameReading(
            at=at,
            screen=screen,
            deck_remaining=self._deck(frame, area, refusals),
            coins=self._coins(frame, area, refusals),
            hand=self._hand(frame, area, refusals),
            roster=roster,
            bonus=self._bonus(frame, area, refusals),
            refusals=refusals,
        )

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

    def _hand(self, frame, area, refusals) -> tuple[Card, ...]:
        row = find_row(area.crop(frame, layout.HAND))
        if row is None:
            refusals["hand"] = "no row of cards in the hand region"
            return ()
        cards: list[Card] = []
        for index, card in enumerate(row.cards):
            match = self.templates.identify(card)
            colour, _ = classify_colour(frame_colour(card))
            if match.character is None:
                refusals[f"hand_{index}"] = match.reason
            if colour is None:
                refusals[f"hand_{index}_colour"] = "frame colour is not blue, orange or pink"
            cards.append(Card(match.character, colour, match.score, match.margin))
        return tuple(cards)

    def _bonus(self, frame, area, refusals) -> str | None:
        row = find_row(area.crop(frame, layout.BONUS_CARD), expected=1)
        if row is None or not row.cards:
            refusals["bonus"] = "no card in the bonus region"
            return None
        match = self.templates.identify(row.cards[0])
        if match.character is None:
            refusals["bonus"] = match.reason
        return match.character

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
