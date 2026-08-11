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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from .reader import FrameReading

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
    refusals: dict[str, str] = field(default_factory=dict)
    crops: dict[str, str] = field(default_factory=dict)

    @classmethod
    def of(cls, reading: FrameReading, *, round_index: int,
           crops: dict[str, str] | None = None) -> "Record":
        return cls(
            at=reading.at,
            round_index=round_index,
            screen=reading.screen,
            deck_remaining=reading.deck_remaining,
            coins=dict(reading.coins),
            hand=[str(card) for card in reading.hand],
            roster=list(reading.roster) if reading.roster else None,
            bonus=reading.bonus,
            refusals=dict(reading.refusals),
            crops=dict(crops or {}),
        )


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
            if self.deck_low is not None and deck > self.deck_low + NEW_DEAL_RISE:
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
                yield Record(**json.loads(line))

    def __len__(self) -> int:
        return sum(1 for _ in self.records())
