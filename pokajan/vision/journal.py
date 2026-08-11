"""The record of what was seen, so the screenshots do not have to be kept.

A frame is grabbed, read, and dropped. What survives is a line in here -- which means this
file *is* the game history, and anything not written down is gone. That shapes two decisions.

**Refusals are recorded, not dropped.** A history with silent gaps is worse than one with
holes in it, because the holes are what tell a later reader that a turn was missed rather
than that nothing happened. Every record carries the reasons for whatever it could not read.

**A round is inferred, not announced.** The game does not label rounds, so a boundary is
detected from two things it cannot fake: the roster changing, and the deck counter going *up*
-- which only a fresh deal can do. Both matter. The roster is the stronger signal but is
unreadable while a payout covers the panel, and a payout is exactly what tends to be on
screen when a round ends.

Records are JSON Lines under `data/games/`, which `.gitignore` already excludes: the carve-out
there is for `*.yaml` and `*.md`, so a `.jsonl` is local by default. That is the right default
-- this is a log of specific online matches, unlike the derived *text* in `data/captures/`
that is committed on purpose.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Iterator

from .reader import DECK_BEFORE_DEAL, Card, FrameReading, Meld

GAMES_DIR = Path(__file__).resolve().parents[2] / "data" / "games"

# A deck counter that has gone up by more than this is a new deal rather than a misread.
# The counter only ever falls within a round, so any rise is suspicious -- but a single digit
# misread can invent one, so the jump has to be big enough to mean something. The opening
# leaves 72 and a round is over well below that.
NEW_DEAL_RISE = 10


@dataclass(frozen=True)
class Record:
    """One settled frame, as it will be remembered."""

    at: float
    round_index: int
    screen: str
    deck_remaining: int | None = None
    coins: dict[str, int] = field(default_factory=dict)
    hand: list[str] = field(default_factory=list)
    roster: list[str] | None = None
    bonus: str | None = None
    bonus_best: str | None = None
    discards: dict[str, list[str]] = field(default_factory=dict)
    melds: list[dict] = field(default_factory=list)
    current_seat: str | None = None
    # Where frame-coloured things sat in the middle of the table, on the frames where a payout
    # was animating. Scaffolding for one open question -- where the meld is drawn for callers
    # other than the right seat -- and it goes when that is answered. Text, not pixels: the
    # game-over banner puts a player's name across this very region.
    survey: list[str] = field(default_factory=list)
    refusals: dict[str, str] = field(default_factory=dict)
    crops: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, reading: FrameReading, *, round_index: int,
           crops: dict[str, str] | None = None,
           survey: tuple[str, ...] | list[str] = ()) -> "Record":
        return cls(
            at=reading.at,
            round_index=round_index,
            screen=reading.screen,
            deck_remaining=reading.deck_remaining,
            coins=dict(reading.coins),
            hand=[str(card) for card in reading.hand],
            roster=list(reading.roster) if reading.roster else None,
            bonus=reading.bonus,
            bonus_best=reading.bonus_best,
            discards={seat: [str(card) for card in cards]
                      for seat, cards in reading.discards.items()},
            melds=[{"seat": m.seat, "cards": [str(c) for c in m.cards]}
                   for m in reading.melds],
            current_seat=reading.current_seat,
            survey=list(survey),
            refusals=dict(reading.refusals),
            crops=dict(crops or {}),
        )

    def reading(self) -> FrameReading:
        """Back into a `FrameReading`, so a logged round replays through the accumulator.

        The round trip is deliberately lossy in one direction only: match scores and margins
        are not written down, because a log is a record of what was concluded and not of how
        narrowly. Everything the accumulator reads is preserved exactly, which is what makes a
        logged round a test case for the advisor rather than only a record of one.
        """
        return FrameReading(
            at=self.at,
            screen=self.screen,
            deck_remaining=self.deck_remaining,
            coins=dict(self.coins),
            hand=tuple(_card(text) for text in self.hand),
            roster=tuple(self.roster) if self.roster else None,
            bonus=self.bonus,
            bonus_best=self.bonus_best,
            discards={seat: tuple(_card(text) for text in cards)
                      for seat, cards in self.discards.items()},
            melds=tuple(Meld(m["seat"], tuple(_card(t) for t in m["cards"]))
                        for m in self.melds),
            current_seat=self.current_seat,
            refusals=dict(self.refusals),
        )


    @classmethod
    def parse(cls, raw: dict) -> "Record":
        """One logged line, read back tolerantly.

        The log is append-only and it *is* the history: rounds recorded weeks ago have to stay
        replayable through whatever the accumulator has become since, or the evidence base
        shrinks every time a field is renamed. So a field this version does not know is dropped
        rather than raising, and renames are migrated here.

        `meld` -> `melds` is the first of those. It held one meld; the field now holds a
        candidate per (seat, length). The old lines get `right`, and that is recovered rather
        than guessed: the reader that wrote them had exactly one box, `Box(0.512, 0.364, 0.683,
        0.513)`, which is the right seat's position. A meld from any other seat could not have
        landed in it. Leaving the seat unknown instead was tried and silently threw away three
        confirmed melds from one round, because a candidate has to match the winning seat --
        and all three of those payouts were won by `right`, as the same lines record.
        """
        raw = dict(raw)
        old = raw.pop("meld", None)
        if old and not raw.get("melds"):
            raw["melds"] = [{"seat": "right", "cards": list(old)}]
        if raw.get("bonus_best") is None:
            raw["bonus_best"] = raw.get("bonus") or _best_from_reason(
                (raw.get("refusals") or {}).get("bonus", ""))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


_BEST_MATCH = re.compile(r"^(?:best match (?P<a>\S+) |(?P<b>\S+) and \S+ are )")


def _best_from_reason(reason: str) -> str | None:
    """The holomem a refusal named, recovered from lines written before `bonus_best` existed.

    Reading prose back is not a habit to get into, and it earns its place once: a whole logged
    round refused the bonus on all 86 frames while naming the same holomem on 68 of them, and
    that round is unadvisable without it. The evidence really is in those lines -- the reader
    wrote down what it saw and only the field to hold it was missing.

    Applies to nothing written from here on, since `bonus_best` is recorded directly.
    """
    found = _BEST_MATCH.match(reason or "")
    return (found.group("a") or found.group("b")) if found else None


def _card(text: str) -> Card:
    """`character:colour` back into a Card, with `?` meaning refused."""
    character, _, colour = text.partition(":")
    return Card(None if character == "?" else character,
                None if colour == "?" else colour)


@dataclass
class RoundTracker:
    """Which round a frame belongs to, inferred from what the frame shows.

    Starts at round 0 and advances on evidence. Never *retreats*, because a wrongly split
    round is recoverable by joining two logs while a wrongly joined one silently mixes two
    different decks -- and the deck is what the whole belief is inferred against.
    """

    round_index: int = 0
    roster: tuple[str, ...] | None = None
    deck_low: int | None = None

    def observe(self, reading: FrameReading) -> int:
        """The round this frame belongs to, advancing the count if it is a new one."""
        fresh = False

        if reading.roster is not None:
            if self.roster is not None and reading.roster != self.roster:
                fresh = True
            self.roster = reading.roster

        deck = reading.deck_remaining
        if deck is not None:
            # A rise *and* a plausible fresh deck. The second half is not belt-and-braces: a
            # payout display partly covering the counter turned 65 into a confident 1, and the
            # rise back to 65 then read as a new deal. That reset the belief's coins to the
            # opening 1000 mid-round and cost tracking for the next 56 frames -- one misread
            # digit, a whole round unadvisable. A deal leaves `DECK_BEFORE_DEAL` cards and the
            # counter shows the full deck before it, so anything below that is not a deal
            # however far it rose.
            if (self.deck_low is not None and deck > self.deck_low + NEW_DEAL_RISE
                    and deck >= DECK_BEFORE_DEAL):
                fresh = True
            self.deck_low = deck if fresh or self.deck_low is None else min(self.deck_low, deck)

        if fresh:
            self.round_index += 1
            if reading.roster is not None:
                self.roster = reading.roster
        return self.round_index


class Journal:
    """Append-only JSON Lines. One line per settled frame, and no images anywhere."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @classmethod
    def for_session(cls, stamp: str, directory: str | Path = GAMES_DIR) -> "Journal":
        return cls(Path(directory) / f"{stamp}.jsonl")

    def append(self, record: Record) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def records(self) -> Iterator[Record]:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield Record.parse(json.loads(line))

    def __len__(self) -> int:
        return sum(1 for _ in self.records())
