"""Turning a belief particle into a game you can actually play forward.

Search needs a concrete world: real opponent hands, a real deck in a real order.
The belief supplies hypotheses about the first two, and the order is simply
guessed. What matters is where the raw material comes from.

**Determinization is built from `PublicState`, never from the live engine.** The
temptation is obvious — the engine is right there, it already holds a valid state,
and cloning it would be one line. But an agent that searches by cloning the engine
can only ever play against a simulator. At M8 the opponent is the real game and
there is no engine to clone; all the bot has is a `PublicState` reconstructed from
pixels. Anything that cannot search from that is not the agent this project is
building. So this module reconstructs a playable `GameState` from exactly what a
seat may legitimately see, plus a hypothesis about what it may not.

One piece of context genuinely is not in `PublicState`, and it is worth naming.
Mid-chain, the engine tracks whether the seat still owes a discard, which depends
on whether the chain began with a claim or with an in-turn call. That is not
public — but it is not hidden either: it is a fact about *the searching seat's own
recent actions*, which the agent knows because it took them. It is threaded in as
`chain_from_claim` rather than inferred, because guessing it wrong would silently
mis-simulate every chain.
"""

from __future__ import annotations

import random

from ..core.actions import DecisionType
from ..core.engine import Engine
from ..core.evaluate import can_call_using
from ..core.rules import Rules
from ..core.state import GameState, Phase
from ..server.protocol import PublicState
from .belief import Particle

_PHASE_FOR = {
    DecisionType.DISCARD: Phase.AWAIT_DISCARD,
    DecisionType.IN_TURN_CALL: Phase.AWAIT_IN_TURN_CALL,
    DecisionType.CHAIN: Phase.AWAIT_CHAIN,
    DecisionType.CLAIM: Phase.AWAIT_CLAIMS,
}


def determinize(
    rules: Rules,
    state: PublicState,
    particle: Particle,
    *,
    seat: int,
    decision: DecisionType,
    claimable_slot: int | None = None,
    chain_from_claim: bool = False,
    rng: random.Random | None = None,
) -> Engine:
    """A playable engine positioned at exactly the decision `seat` is facing.

    The returned engine is paused at the same point the real game is, so applying
    a candidate action and rolling forward measures that action and nothing else.
    """
    rng = rng or random.Random()
    players = len(state.seats)

    deck = [slot for slot, n in enumerate(particle.deck) for _ in range(n)]
    rng.shuffle(deck)

    game = GameState(
        rules=rules,
        game_id=state.game_id,
        bonus_character=state.bonus_character,
        deck=deck,
        hands=[hand[:] for hand in particle.hands],
        coins=[view.coins for view in state.seats],
        table=list(state.table),
        scored=list(state.scored),
        discards=[list(view.discards) for view in state.seats],
        recent_discards=[list(r) for r in state.recent_discards],
        calls_made=[view.calls_made for view in state.seats],
        coins_won=[view.coins_won for view in state.seats],
        coins_paid=[view.coins_paid for view in state.seats],
        current_seat=state.current_seat,
        turn_index=state.turn_index,
        phase=_PHASE_FOR[decision],
        coins_minted=state.coins_minted,
        last_discard_slot=state.last_discard_slot,
        last_discard_seat=state.last_discard_seat,
        rng=rng,
    )

    if decision in (DecisionType.DISCARD, DecisionType.IN_TURN_CALL):
        game.current_seat = seat
    elif decision is DecisionType.CHAIN:
        game.chain_seat = seat
        game.chain_from_claim = chain_from_claim
        game.chain_needs_discard = (
            rules.play.discard_after_claim
            if chain_from_claim
            else rules.play.discard_after_in_turn_call
        )

    engine = Engine(game)

    if decision is DecisionType.CLAIM:
        _open_hypothetical_claim_window(engine, state, seat, claimable_slot)

    return engine


def _open_hypothetical_claim_window(
    engine: Engine, state: PublicState, seat: int, slot: int | None
) -> None:
    """Reopen the claim window the searching seat is sitting inside.

    Who else is eligible is not knowable — it depends on hands this seat cannot
    see — so it is *derived from the hypothesis*, exactly as the real engine
    derives it from the real hands. That is the point of determinizing: the search
    gets to find out that a rival could have snatched the card in this world and
    not in that one, which is precisely the risk a claim decision turns on.

    The searching seat is always included. It was asked, so the engine had already
    ruled it eligible, whatever the hypothesis says.
    """
    game = engine.state
    if slot is None:
        raise ValueError("a CLAIM decision requires the claimable slot")

    eligible = []
    for other in engine.claim_priority_order(state.last_discard_seat):
        if other == seat:
            eligible.append(other)
            continue
        probe = game.hands[other][:]
        probe[slot] += 1
        if can_call_using(game.rules, probe, slot, bonus_character=game.bonus_character):
            eligible.append(other)

    game.claim_eligible = eligible
    game.claim_responses = {}
    game.phase = Phase.AWAIT_CLAIMS
