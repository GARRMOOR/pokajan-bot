"""The action space, and the legality masks that go with it.

Layout, for a card space of `n` slots:

    0 .. n-1     DISCARD slot i
    n            CALL     — score the best hand available (in turn, or claiming a discard)
    n+1          PASS     — decline to claim, or decline to continue a chain

`CALL` deliberately does not name *which* hand to score. Only one hand may score
at a time and the tiebreak is by payout, so choosing anything other than the
highest-paying call is never better under the rules — leaving it to the engine
keeps the action space flat and small. (Which *copies* a mixed hand spends is a
separate question, handled through `evaluate.enumerate_calls`, not here.)

Discarding is indexed by slot rather than by position in hand. Two blue AzKi cards
are interchangeable, so a position-indexed action space would make the network
learn that they are the same thing — wasted capacity, and it would break the
moment a hand got sorted differently.
"""

from __future__ import annotations

from enum import IntEnum

from .cards import Counts
from .evaluate import can_call
from .rules import Rules


class DecisionType(IntEnum):
    """Why a seat is being asked to act.

    The engine drives play by asking whichever seat must decide next, rather than
    stepping seats in a fixed order — out-of-turn claims make a fixed order
    impossible.
    """

    DISCARD = 0        # your turn, you have drawn, now throw one
    CLAIM = 1          # someone discarded; claim it or pass
    IN_TURN_CALL = 2   # your turn and your hand already scores; call or play on
    CHAIN = 3          # you just scored and refilled, and can score again


def action_space_size(rules: Rules) -> int:
    return rules.cards.n_slots + 2


def call_action(rules: Rules) -> int:
    return rules.cards.n_slots


def pass_action(rules: Rules) -> int:
    return rules.cards.n_slots + 1


def is_discard(rules: Rules, action: int) -> bool:
    return 0 <= action < rules.cards.n_slots


def legal_mask(
    rules: Rules,
    decision: DecisionType,
    hand: Counts,
    *,
    bonus_character: int | None,
    claimable_slot: int | None = None,
) -> list[bool]:
    """Which actions are legal for this decision.

    Returned separately from the observation: it is a hard constraint the policy
    is masked with, not a feature it should have to infer.
    """
    n = rules.cards.n_slots
    mask = [False] * (n + 2)

    if decision is DecisionType.DISCARD:
        # Any card actually held may be thrown.
        for slot in range(n):
            if hand[slot] > 0:
                mask[slot] = True
        return mask

    if decision is DecisionType.CLAIM:
        # Claiming is only offered when the claimed card would genuinely complete
        # a hand — the engine checks against hand + the discard.
        if claimable_slot is None:
            raise ValueError("CLAIM decision requires claimable_slot")
        probe = list(hand)
        probe[claimable_slot] += 1
        if can_call(rules, probe, bonus_character=bonus_character):
            mask[n] = True
        mask[n + 1] = True     # passing is always allowed
        return mask

    if decision in (DecisionType.IN_TURN_CALL, DecisionType.CHAIN):
        if can_call(rules, hand, bonus_character=bonus_character):
            mask[n] = True
        # Declining is always allowed: chaining is optional, and there are real
        # positions where you would rather not — every refill drains the shared
        # deck, and running it dry ends the game.
        mask[n + 1] = True
        return mask

    raise ValueError(f"unhandled decision type {decision!r}")


def describe_action(rules: Rules, action: int) -> str:
    n = rules.cards.n_slots
    if action < n:
        return f"discard {rules.cards.describe_slot(action)}"
    if action == n:
        return "call Pokajan"
    return "pass"
