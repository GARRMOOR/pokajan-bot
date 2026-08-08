"""The agent interface.

An agent is anything that can answer a DecisionRequest. The engine does not care
whether the answer comes from a heuristic, a neural network, a Monte-Carlo search,
or a human clicking in a browser — which is what lets the same evaluation harness
run all four against each other.
"""

from __future__ import annotations

import random
from typing import Protocol

from ..core.actions import call_action, pass_action
from ..core.rules import Rules
from ..server.protocol import ActionResponse, DecisionRequest


class Agent(Protocol):
    name: str

    def act(self, request: DecisionRequest) -> ActionResponse:
        ...


def legal_actions(request: DecisionRequest) -> list[int]:
    return [i for i, ok in enumerate(request.legal_mask) if ok]


class RandomAgent:
    """Uniform over legal actions.

    Not a serious opponent, but the one that finds engine bugs: it will happily
    chain calls until the deck dies, claim hands nobody would claim, and discard
    cards it just drew, which is exactly the coverage the invariant tests want.
    """

    def __init__(self, seed: int | None = None, name: str = "random") -> None:
        self.rng = random.Random(seed)
        self.name = name

    def act(self, request: DecisionRequest) -> ActionResponse:
        return ActionResponse(
            game_id=request.game_id,
            seat=request.seat,
            action=self.rng.choice(legal_actions(request)),
        )


class GreedyCallerAgent:
    """Always calls when it can; otherwise discards at random.

    A useful reference point rather than a good strategy — it establishes what
    happens when nobody ever declines a chain, which is the fastest route to both
    end conditions and so exercises them hard.
    """

    def __init__(self, rules: Rules, seed: int | None = None, name: str = "greedy") -> None:
        self.rules = rules
        self.rng = random.Random(seed)
        self.name = name

    def act(self, request: DecisionRequest) -> ActionResponse:
        call = call_action(self.rules)
        if request.legal_mask[call]:
            action = call
        else:
            options = [a for a in legal_actions(request) if a != pass_action(self.rules)]
            action = self.rng.choice(options or legal_actions(request))
        return ActionResponse(game_id=request.game_id, seat=request.seat, action=action)
