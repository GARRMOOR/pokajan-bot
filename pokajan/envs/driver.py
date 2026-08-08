"""Running whole games: the loop that sits between an Engine and some agents.

Kept separate from the engine because it is the piece that changes shape most
often — self-play, evaluation with seat rotation, a human at one seat, and
Monte-Carlo rollouts all want the same engine driven slightly differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.engine import Engine
from ..core.rules import Rules


@dataclass
class GameResult:
    game_id: str
    final_coins: list[int]
    standings: list[int]
    end_reason: str | None
    turns: int
    decisions: int
    coins_minted: int
    calls_made: list[int]
    deck_remaining: int
    history: list[dict] = field(default_factory=list)

    @property
    def winner(self) -> int:
        return self.standings[0]

    def coin_delta(self, seat: int, initial: int) -> int:
        return self.final_coins[seat] - initial


def play_game(
    engine: Engine,
    agents: list,
    *,
    max_decisions: int = 100_000,
    record_history: bool = False,
) -> GameResult:
    """Drive one game to completion.

    The claim window is the reason this asks for *all* pending decisions and
    answers them as a batch rather than looping one seat at a time: every eligible
    seat must decide against the same state, without seeing what the others did.
    """
    decisions = 0
    history: list[dict] = []

    while not engine.finished:
        pending = engine.pending_decisions()
        if not pending:
            break

        responses = []
        for request in pending:
            response = agents[request.seat].act(request)
            if response.seat != request.seat:
                raise ValueError(
                    f"agent for seat {request.seat} returned a response for seat {response.seat}"
                )
            responses.append(response)
            decisions += 1
            if record_history:
                history.append(
                    {
                        "turn": request.state.turn_index,
                        "seat": request.seat,
                        "decision": request.decision,
                        "action": response.action,
                        "coins": [s.coins for s in request.state.seats],
                    }
                )

        engine.submit(responses)

        if decisions > max_decisions:
            raise RuntimeError(
                f"game {engine.state.game_id} exceeded {max_decisions} decisions — "
                "this means play is not progressing, not that the game is long"
            )

    s = engine.state
    return GameResult(
        game_id=s.game_id,
        final_coins=engine.final_coins(),
        standings=engine.standings(),
        end_reason=s.end_reason,
        turns=s.turn_index,
        decisions=decisions,
        coins_minted=s.coins_minted,
        calls_made=s.calls_made[:],
        deck_remaining=s.deck_remaining,
        history=history,
    )


def play_many(
    rules: Rules,
    agent_factory,
    *,
    games: int = 100,
    seed: int = 0,
) -> list[GameResult]:
    """Play `games` independent games. `agent_factory(seed)` builds one seat list."""
    results = []
    for i in range(games):
        engine = Engine.new_game(rules, seed=seed + i)
        results.append(play_game(engine, agent_factory(seed + i)))
    return results
