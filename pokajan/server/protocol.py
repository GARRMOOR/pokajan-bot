"""The wire format between "a Pokajan game" and "something that plays it".

This is the seam the whole project hangs off, so it is worth being fussy about.
Four different things produce or consume these messages:

  * the simulator, when a human plays in the browser (M2)
  * the replay viewer, reading logged games (M2)
  * the AI hint panel, asking for a recommendation (M7)
  * the screen reader (M8), which does not simulate anything — it *watches the
    real game* and emits a PublicState

That last one is why this module exists this early. If the recommender only ever
talks PublicState, then reading the real game is a matter of building one of these
from pixels, and nothing else in the stack has to know the difference.

Hence the hard rule: **PublicState contains only what a seat may legitimately
see.** No opponent hands, no deck order. Anything a screen reader could not
observe must not appear here, or the AI will quietly learn to depend on
information it will not have when it matters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class GameSetup:
    """Static for a whole game — sent once, not per decision."""

    protocol_version: int
    rules_hash: str
    rules_name: str
    game_id: str

    character_ids: list[str]
    character_names: list[str]
    colors: list[str]
    group_ids: list[str]
    group_names: list[str]
    group_members: list[list[int]]     # character indices per group

    n_slots: int
    hand_limit: int
    players: int
    initial_coins: int
    deck_size: int
    bonus_character: int | None

    def slot_label(self, slot: int) -> str:
        n_colors = len(self.colors)
        return f"{self.character_names[slot // n_colors]} ({self.colors[slot % n_colors]})"


@dataclass(frozen=True)
class SeatView:
    """What is publicly known about one seat."""

    seat: int
    coins: int
    hand_size: int
    discards: list[int]          # count vector of everything this seat has thrown
    calls_made: int
    coins_won: int
    coins_paid: int


@dataclass(frozen=True)
class PublicState:
    """One seat's complete, legitimate view of the game."""

    game_id: str
    turn_index: int
    viewer: int                  # which seat this view belongs to

    hand: list[int]              # the viewer's own hand, as a count vector
    seats: list[SeatView]

    deck_remaining: int          # a count only — never the contents or order
    table: list[int]             # every card discarded so far, unclaimable
    scored: list[int]            # every card removed by a scored hand

    last_discard_slot: int | None
    last_discard_seat: int | None

    current_seat: int
    phase: str
    finished: bool = False
    final_coins: list[int] | None = None
    coins_minted: int = 0        # from the bankruptcy floor; see rules.yaml

    # Also on GameSetup, and repeated here on purpose. The bonus holomem is drawn
    # per game and displayed by the real game, so it is legitimately public — and
    # an agent that only ever sees DecisionRequests cannot price a hand without
    # it. Keeping PublicState self-sufficient is what lets agents stay stateless.
    bonus_character: int | None = None

    # Recent discard order per seat, most recent first. Redundant with `discards`
    # but the ordering carries the read on what a player is collecting.
    recent_discards: list[list[int]] = field(default_factory=list)


@dataclass(frozen=True)
class DecisionRequest:
    """The engine asking one seat to act."""

    game_id: str
    seat: int
    decision: str                # DecisionType name: DISCARD | CLAIM | IN_TURN_CALL | CHAIN
    legal_mask: list[bool]
    state: PublicState

    claimable_slot: int | None = None
    # What CALL would score right now, for the GUI and for hint display. Advisory:
    # the engine recomputes it rather than trusting a client.
    best_call_payout: int | None = None
    best_call_label: str | None = None


@dataclass(frozen=True)
class ActionResponse:
    """A seat's answer."""

    game_id: str
    seat: int
    action: int

    # Optional override of which copies a mixed-colour hand spends. None means the
    # engine's canonical selection. Ignored for anything but CALL.
    card_selection: list[int] | None = None


@dataclass(frozen=True)
class Recommendation:
    """A hint for the human: what the bot would do, and how sure it is.

    Carries the risk setting it was produced under, because the same position has
    genuinely different right answers at max-EV versus safe.
    """

    seat: int
    action: int
    action_label: str
    confidence: float
    risk_alpha: float
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    expected_final_coins: float | None = None
    p_finish_above_start: float | None = None
    reasoning: str | None = None


def to_json(message: Any) -> str:
    return json.dumps(asdict(message), separators=(",", ":"))


def to_dict(message: Any) -> dict[str, Any]:
    return asdict(message)
