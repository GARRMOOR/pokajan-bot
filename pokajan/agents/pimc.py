"""Perfect-Information Monte Carlo: decide by playing out worlds you might be in.

The idea is simple enough to state in a sentence. Sample whole hypotheses about
the hidden cards from the belief, and in each one play every candidate action
forward to the end of the game with a fast policy. Whichever action wins the most
coins averaged over hypotheses is the one to play.

Three details do most of the work:

**Every action faces the same worlds.** Candidates are compared inside a single
determinization before moving to the next, so a lucky deck helps all of them
equally. Without that pairing the noise between actions swamps the difference
between them — this game swings by more than the starting stack on one hand.

**Candidates are shortlisted first.** A discard has seven options and evaluating
all of them costs seven times as much as evaluating the two that could plausibly
be right. The heuristic ranks them, the search settles between the top few, and
the particles it sampled to do the ranking are the same ones the search plays out.

**Search only happens where it can change the answer.** With one legal action there
is nothing to decide, and the whole apparatus is skipped.

### What this does not do

PIMC has a known and genuine weakness, and it is worth stating rather than
discovering later. Inside each determinization every player sees every card, so
the simulated opponents never have to guess and this agent never gains from making
them guess. It systematically undervalues concealment, and it will happily assume
an opponent finds the one defence that requires knowing your hand. The fix is
search over information sets rather than states, which is a different and far more
expensive algorithm — the reason the roadmap goes to learned policies at M5 rather
than to deeper search.

Costs, measured rather than predicted: about 8ms for a full rollout from the deal,
so a decision at the default settings lands near 100-150ms. That is comfortable for
a live hint at M7 and for labelling behaviour-cloning data at M5, and far too slow
for a PPO inner loop — which is exactly the split the roadmap assumes.
"""

from __future__ import annotations

import random

from ..core.actions import DecisionType, call_action, pass_action
from ..core.rules import Rules
from ..envs.determinize import determinize
from ..server.protocol import ActionResponse, DecisionRequest, PublicState
from .heuristic import HeuristicAgent
from .rollout import RolloutPolicy, play_out

DEFAULT_DETERMINIZATIONS = 16
DEFAULT_CANDIDATES = 3


class PIMCAgent:
    """Searches determinized worlds, guided by the heuristic and the belief."""

    def __init__(
        self,
        rules: Rules,
        *,
        determinizations: int = DEFAULT_DETERMINIZATIONS,
        candidates: int = DEFAULT_CANDIDATES,
        rollout_depth: int = 400,
        search_discards: bool = True,
        seed: int | None = None,
        name: str = "pimc",
    ) -> None:
        self.rules = rules
        self.determinizations = determinizations
        self.candidates = max(2, candidates)
        self.rollout_depth = rollout_depth
        self.search_discards = search_discards
        self.name = name
        self.rng = random.Random(seed)

        # The prior owns the belief. PIMC does not keep a second one: the particles
        # that rank the candidates are the same particles the search plays out, so
        # two beliefs would mean sampling twice and disagreeing about the world.
        self.prior = HeuristicAgent(rules, seed=seed, particles=determinizations)
        self.policy = RolloutPolicy(rules, seed=seed)

        self._call = call_action(rules)
        self._pass = pass_action(rules)
        self._chain_from_claim = False

    # ------------------------------------------------------------- acting ---
    def act(self, request: DecisionRequest) -> ActionResponse:
        state = request.state
        decision = DecisionType[request.decision]

        self.prior.start_game(state)
        self.prior.belief.observe(state)

        legal = [action for action, ok in enumerate(request.legal_mask) if ok]
        if len(legal) == 1:
            return self._respond(request, legal[0], decision)

        particles = self.prior.belief.sample(self.determinizations)
        shortlist = self._shortlist(state, decision, legal, particles)
        if len(shortlist) == 1:
            return self._respond(request, shortlist[0], decision)

        totals = {action: 0.0 for action in shortlist}
        initial = self.rules.play.initial_coins
        seat = request.seat

        for particle in particles:
            base = determinize(
                self.rules,
                state,
                particle,
                seat=seat,
                decision=decision,
                claimable_slot=request.claimable_slot,
                chain_from_claim=self._chain_from_claim,
                rng=self.rng,
            )
            for action in shortlist:
                fork = base.clone()
                fork.apply([(seat, action)])
                coins = play_out(fork, self.policy, max_rounds=self.rollout_depth)
                totals[action] += particle.weight * (coins[seat] - initial)

        # Ties resolve toward the shortlist order, which is the heuristic's own
        # ranking — so search only overrules the prior when it has a reason to.
        best = max(shortlist, key=lambda action: (totals[action], -shortlist.index(action)))
        return self._respond(request, best, decision)

    def _shortlist(
        self, state: PublicState, decision: DecisionType, legal: list[int], particles
    ) -> list[int]:
        """The few actions worth spending rollouts on, best-first.

        Call-or-pass decisions are already down to two, and they are the ones where
        search earns most — declining a chain to keep the deck alive, or refusing a
        small claim that would break a big hand. Only discards need pruning.
        """
        if decision is not DecisionType.DISCARD:
            return legal
        if not self.search_discards:
            # Hand the whole discard decision back to the heuristic — see the note
            # on rollout variance in the class docstring.
            return [self.prior.act_discard(state, particles)]

        context = self.prior.valuation_context(state)
        danger = self.prior.danger(legal, particles, state)
        scored = []
        for slot in legal:
            trial = list(state.hand)
            trial[slot] -= 1
            scored.append((self.prior.hand_value(trial, context) - danger.get(slot, 0.0), slot))
        scored.sort(reverse=True)
        return [slot for _, slot in scored[: self.candidates]]

    def _respond(
        self, request: DecisionRequest, action: int, decision: DecisionType
    ) -> ActionResponse:
        # A chain refills you to the size you were at before calling, which differs
        # depending on how the chain started. That is not in PublicState — but it is
        # a fact about this seat's own last move, so it is remembered rather than
        # guessed. Getting it wrong mis-simulates every chain the search runs.
        if decision is DecisionType.CLAIM:
            self._chain_from_claim = action == self._call
        elif decision is DecisionType.IN_TURN_CALL:
            self._chain_from_claim = False
        elif decision is DecisionType.DISCARD:
            self._chain_from_claim = False

        return ActionResponse(game_id=request.game_id, seat=request.seat, action=action)
