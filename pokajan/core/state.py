"""Mutable game state, deck construction, and the phase enum.

State is deliberately plain data: parallel lists of counts and ints, no objects
holding references to each other. Two reasons.

First, `clone()` has to be cheap. Perfect-information Monte-Carlo search copies the
whole state once per determinization — tens of thousands of times a second — so a
copy is a handful of list slices and nothing deeper.

Second, the engine is an explicit state machine rather than a coroutine. A
generator would read more naturally for something as sequential as a turn, but you
cannot copy a suspended generator, and search needs to fork a game mid-decision.
So every point where play stops to ask someone something is a `Phase` value plus a
few pending-context fields, all of which copy trivially.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, IntEnum

from .cards import Counts
from .rules import Composition, Rules


class Phase(IntEnum):
    """Where play has stopped, and therefore who is being asked what."""

    AWAIT_IN_TURN_CALL = 0   # your turn, your hand already scores: call or play on
    AWAIT_DISCARD = 1        # your turn, throw one
    AWAIT_CLAIMS = 2         # a card was discarded; every eligible seat answers at once
    AWAIT_CHAIN = 3          # you scored and refilled, and can score again
    FINISHED = 4


class EndReason(str, Enum):
    DECK_EMPTY = "deck_empty"
    BANKRUPT = "bankrupt"


def build_deck(rules: Rules, rng: random.Random) -> list[int]:
    """The shuffled draw pile, as a list of slot indices.

    How the 100 cards are chosen from the roster is genuinely unknown — see the
    `composition` TODO in the rules file. Both strategies below are plausible and
    produce noticeably different games, which is why the belief model infers
    composition from play rather than assuming it.
    """
    space = rules.cards
    n_slots = space.n_slots
    cap = space.max_per_color
    size = rules.deck_size

    if rules.composition is Composition.EVEN_AS_POSSIBLE:
        base, remainder = divmod(size, n_slots)
        counts = [base] * n_slots
        # Validation guarantees base <= cap, and that base == cap implies
        # remainder == 0, so every slot chosen here has room for one more.
        for slot in rng.sample(range(n_slots), remainder):
            counts[slot] += 1
    elif rules.composition is Composition.UNIFORM_RANDOM:
        pool = [slot for slot in range(n_slots) for _ in range(cap)]
        rng.shuffle(pool)
        counts = [0] * n_slots
        for slot in pool[:size]:
            counts[slot] += 1
    else:  # pragma: no cover - Composition is exhaustive
        raise ValueError(f"unhandled composition {rules.composition!r}")

    deck = [slot for slot, n in enumerate(counts) for _ in range(n)]
    rng.shuffle(deck)
    return deck


@dataclass
class GameState:
    """Everything about one game in progress."""

    rules: Rules
    game_id: str
    bonus_character: int | None

    deck: list[int]                    # draw pile; cards are taken from the end
    hands: list[Counts]
    coins: list[int]

    table: Counts                      # every discard that was not claimed
    scored: Counts                     # every card removed by a scored hand
    discards: list[Counts]             # per seat, everything they have thrown
    recent_discards: list[list[int]]   # per seat, most recent first

    calls_made: list[int]
    coins_won: list[int]
    coins_paid: list[int]

    current_seat: int = 0
    turn_index: int = 0
    phase: Phase = Phase.AWAIT_DISCARD
    finished: bool = False
    end_reason: str | None = None

    # Coins created by the bankruptcy floor: a payer stops at zero but the caller
    # still receives the full amount. Tracked so tests can assert exactly how much
    # was minted rather than giving up on coin accounting altogether.
    coins_minted: int = 0

    last_discard_slot: int | None = None
    last_discard_seat: int | None = None

    # --- pending-decision context -------------------------------------------
    # A seat part-way through a chain of calls, and whether it still owes a
    # discard when the chain ends (it does for an in-turn call, not for a claim).
    chain_seat: int | None = None
    chain_needs_discard: bool = False
    chain_from_claim: bool = False

    # The open claim window. Every eligible seat is asked against this same state,
    # and nothing mutates until all of them have answered — that is what stops one
    # seat's decision leaking into another's.
    claim_eligible: list[int] = field(default_factory=list)
    claim_responses: dict[int, int] = field(default_factory=dict)

    rng: random.Random = field(default_factory=random.Random)

    # ---------------------------------------------------------------- misc --
    @property
    def players(self) -> int:
        return self.rules.play.players

    @property
    def deck_remaining(self) -> int:
        return len(self.deck)

    def hand_size(self, seat: int) -> int:
        return sum(self.hands[seat])

    def clone(self) -> "GameState":
        """A copy that can be played forward independently.

        `rules` is shared rather than copied: it is frozen, and copying the card
        space per determinization would dominate the cost of search.
        """
        copy = GameState(
            rules=self.rules,
            game_id=self.game_id,
            bonus_character=self.bonus_character,
            deck=self.deck[:],
            hands=[h[:] for h in self.hands],
            coins=self.coins[:],
            table=self.table[:],
            scored=self.scored[:],
            discards=[d[:] for d in self.discards],
            recent_discards=[r[:] for r in self.recent_discards],
            calls_made=self.calls_made[:],
            coins_won=self.coins_won[:],
            coins_paid=self.coins_paid[:],
            current_seat=self.current_seat,
            turn_index=self.turn_index,
            phase=self.phase,
            finished=self.finished,
            end_reason=self.end_reason,
            coins_minted=self.coins_minted,
            last_discard_slot=self.last_discard_slot,
            last_discard_seat=self.last_discard_seat,
            chain_seat=self.chain_seat,
            chain_needs_discard=self.chain_needs_discard,
            chain_from_claim=self.chain_from_claim,
            claim_eligible=self.claim_eligible[:],
            claim_responses=dict(self.claim_responses),
            rng=random.Random(),
        )
        copy.rng.setstate(self.rng.getstate())
        return copy

    def cards_in_play(self) -> int:
        """Total cards accounted for — the card-conservation invariant.

        Should always equal the deck size: nothing is created and nothing is
        destroyed, cards only move between the deck, hands, the table and the
        scored pile.
        """
        return (
            len(self.deck)
            + sum(sum(h) for h in self.hands)
            + sum(self.table)
            + sum(self.scored)
        )
