"""A PettingZoo-style AEC wrapper over the engine.

The standard single-agent Gym loop does not fit this game: after a discard, three
other seats may all have something to say before play continues. AEC does fit —
it asks whichever agent is up next, one at a time — so that is the shape here.

The subtlety is the claim window, where several seats must decide *without seeing
each other's answers*. Handing them out one at a time looks like it would leak
that information, but it does not: the engine accumulates claim responses and
mutates nothing until every eligible seat has answered, and it rebuilds each
request from that same untouched state. So the adapter can stay simple, and the
guarantee stays in the engine where it can be tested directly.

Deliberately dependency-free. PettingZoo is not installed, and the training loop
drives the engine directly for speed — this exists so standard tooling and
human-readable rollouts have a familiar interface.
"""

from __future__ import annotations

from ..core.engine import Engine
from ..core.rules import Rules
from ..server.protocol import ActionResponse, DecisionRequest


class PokajanAECEnv:
    """Agent-Environment-Cycle interface for one Pokajan game."""

    metadata = {"name": "pokajan_v0", "is_parallelizable": False}

    def __init__(self, rules: Rules, seed: int | None = None) -> None:
        self.rules = rules
        self._seed = seed
        self.engine: Engine | None = None
        self.possible_agents = list(range(rules.play.players))
        self.reset(seed)

    # -------------------------------------------------------------- cycle ---
    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._seed = seed
        self.engine = Engine.new_game(self.rules, seed=self._seed)
        self.agents = list(self.possible_agents)

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._coins = self.engine.state.coins[:]
        self._current: DecisionRequest | None = None
        self._sync()

    @property
    def agent_selection(self) -> int | None:
        """The seat that must act, or None once the game is over."""
        return None if self._current is None else self._current.seat

    def last(self) -> tuple[DecisionRequest | None, float, bool, bool, dict]:
        seat = self.agent_selection
        if seat is None:
            done_seat = self.agents[0]
            return None, 0.0, True, False, self.infos[done_seat]
        return (
            self._current,
            self._cumulative_rewards[seat],
            self.terminations[seat],
            self.truncations[seat],
            self.infos[seat],
        )

    def observe(self, seat: int):
        return self.engine.public_state(seat)

    def step(self, action: int) -> None:
        if self._current is None:
            raise ValueError("game is over; call reset()")

        seat = self._current.seat
        self._cumulative_rewards[seat] = 0.0
        self.engine.submit(
            ActionResponse(game_id=self.engine.state.game_id, seat=seat, action=action)
        )
        self._sync()

    # ------------------------------------------------------------ internals --
    def _sync(self) -> None:
        """Refresh rewards, termination flags, and whoever is up next.

        Rewards are the per-event coin deltas, scaled by the starting stack. They
        sum exactly to the terminal `(final - initial) / initial`, which makes the
        shaping potential-based and therefore unbiased — it moves credit closer to
        the decision that earned it without changing what the optimal policy is.
        """
        engine = self.engine
        coins = engine.state.coins
        scale = float(self.rules.play.initial_coins)

        for seat in self.agents:
            delta = (coins[seat] - self._coins[seat]) / scale
            self.rewards[seat] = delta
            self._cumulative_rewards[seat] += delta
        self._coins = coins[:]

        pending = engine.pending_decisions()
        self._current = pending[0] if pending else None

        if engine.finished:
            for seat in self.agents:
                self.terminations[seat] = True
                self.infos[seat] = {
                    "end_reason": engine.state.end_reason,
                    "final_coins": engine.final_coins(),
                    "standings": engine.standings(),
                    "coins_minted": engine.state.coins_minted,
                }

    # -------------------------------------------------------------- extras ---
    def legal_mask(self) -> list[bool] | None:
        return None if self._current is None else self._current.legal_mask

    def clone(self) -> "PokajanAECEnv":
        """Fork mid-game, for search. Shares rules, copies state."""
        fork = object.__new__(PokajanAECEnv)
        fork.rules = self.rules
        fork._seed = self._seed
        fork.engine = self.engine.clone()
        fork.possible_agents = list(self.possible_agents)
        fork.agents = list(self.agents)
        fork.rewards = dict(self.rewards)
        fork._cumulative_rewards = dict(self._cumulative_rewards)
        fork.terminations = dict(self.terminations)
        fork.truncations = dict(self.truncations)
        fork.infos = {k: dict(v) for k, v in self.infos.items()}
        fork._coins = self._coins[:]
        fork._sync()
        return fork
