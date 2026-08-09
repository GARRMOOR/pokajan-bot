"""Turning a decision into advice a human can act on.

The agents decide; this explains. That is a different job, and the difference
matters more here than in most games, because the reasoning is frequently
counter-intuitive: the right discard is often the one that breaks up your most
valuable-looking hand, because a monochrome triple pays seven times a mixed one
and is correspondingly unlikely, or because the card you were about to throw is
worth 840 to the player on your left.

Advice nobody believes is worthless, so every recommendation carries the two
numbers behind it — what the rest of the hand is worth, and what the discard risks
— rather than a bare instruction.

This is the producer for `protocol.Recommendation`, and it is deliberately the
only one. Both M7 surfaces consume it: the hint panel in the browser and the
overlay pinned over the real game. At M8 the screen reader becomes another source
of `PublicState` feeding the same call, and nothing here changes.
"""

from __future__ import annotations

import math

from ..core.actions import DecisionType, call_action, describe_action, pass_action
from ..core.evaluate import best_call
from ..core.rules import Rules
from ..server.protocol import DecisionRequest, Recommendation
from .heuristic import HeuristicAgent, _spend

# The danger estimate's standard deviation across independent redraws of the
# belief, measured at 48 particles (see the M4 notes in the README). It falls as
# 1/sqrt(n), which is what makes a stated confidence meaningful rather than
# decorative.
DANGER_SD_AT_48 = 14.8
BASELINE_PARTICLES = 48


class Advisor:
    """Wraps an agent and explains what it would do, and why."""

    def __init__(
        self,
        rules: Rules,
        *,
        agent: HeuristicAgent | None = None,
        seed: int | None = None,
        particles: int = BASELINE_PARTICLES,
        risk_alpha: float = 0.0,
    ) -> None:
        self.rules = rules
        self.agent = agent or HeuristicAgent(rules, seed=seed, particles=particles)
        # Carried through onto every recommendation because the same position has
        # genuinely different right answers at max-EV versus safe. Until M6 trains
        # a risk-conditioned policy the agent ignores it, so it is reported rather
        # than applied — saying so is better than implying a setting that does
        # nothing.
        self.risk_alpha = risk_alpha

    # ------------------------------------------------------------- advising --
    def observe(self, state) -> None:
        """Fold a public state into the belief without asking for advice.

        Separate from `recommend` because the belief is *accumulated*, not derived:
        an unclaimed discard is only visible as a difference between two
        consecutive views, so a belief that only ever sees the positions somebody
        happened to ask about has holes in it. Callers should feed every state they
        see and request advice separately.

        This is the same constraint the M8 screen reader lives under, so the
        browser panel and the overlay share it rather than the overlay discovering
        it later.
        """
        self.agent.start_game(state)
        self.agent.belief.observe(state)

    def recommend(self, request: DecisionRequest) -> Recommendation:
        state = request.state
        self.observe(state)

        decision = DecisionType[request.decision]
        if decision is DecisionType.DISCARD:
            return self._advise_discard(request)
        return self._advise_call(request, decision)

    def _advise_discard(self, request: DecisionRequest) -> Recommendation:
        state = request.state
        space = self.agent.space
        hand = state.hand
        candidates = [slot for slot in range(space.n_slots) if hand[slot] > 0]

        particles = self.agent._particles()
        value = self.agent.valuer(state, particles)
        danger = self.agent.danger(candidates, particles, state)

        scored = []
        for slot in candidates:
            trial = hand[:]
            trial[slot] -= 1
            keeps = value(trial)
            risk = danger.get(slot, 0.0)
            scored.append(
                {
                    "action": slot,
                    "label": space.describe_slot(slot),
                    "score": keeps - risk,
                    "hand_value": keeps,
                    "danger": risk,
                }
            )
        scored.sort(key=lambda row: -row["score"])

        best = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None
        gap = 0.0 if runner_up is None else best["score"] - runner_up["score"]
        confidence = self._confidence(gap)

        # How far behind the recommendation each option is, rather than leaving the
        # subtraction to whoever renders this. `score` nets the danger off the hand
        # value, so a panel showing raw scores next to the headline's hand value
        # invites a comparison between two different quantities -- observed doing
        # exactly that, reading a 5-coin gap where the real one was nil.
        for row in scored[1:]:
            row["behind"] = best["score"] - row["score"]

        return Recommendation(
            seat=request.seat,
            action=best["action"],
            action_label=f"discard {best['label']}",
            confidence=confidence,
            risk_alpha=self.risk_alpha,
            alternatives=scored[1:4],
            reasoning=self._explain_discard(best, runner_up, confidence),
        )

    def _advise_call(self, request: DecisionRequest, decision: DecisionType) -> Recommendation:
        """Call, claim or chain — always a choice between banking and keeping."""
        state = request.state
        slot = request.claimable_slot
        probe = state.hand[:]
        if slot is not None:
            probe[slot] += 1

        call = best_call(
            self.rules,
            probe,
            bonus_character=self.agent._bonus,
            claimed=slot is not None,
            must_use=slot,
        )
        take_action = call_action(self.rules)
        skip_action = pass_action(self.rules)

        if call is None:
            return Recommendation(
                seat=request.seat,
                action=skip_action,
                action_label=describe_action(self.rules, skip_action),
                confidence=1.0,
                risk_alpha=self.risk_alpha,
                reasoning="Nothing here scores.",
            )

        value = self.agent.valuer(state, self.agent._particles())
        keep = value(state.hand)
        take = call.payout + value(_spend(probe, call))
        wants_it = take >= keep

        taking = {
            "action": take_action,
            # The hand goes in the label rather than being left to the alternatives
            # list, so the headline reads "call Pokajan: triple Mori Calliope (120)"
            # on its own. The overlay often shows nothing else.
            "label": f"call Pokajan: {call.describe(self.agent.space)}",
            "score": take,
            "payout": call.payout,
        }
        passing = {
            "action": skip_action,
            "label": "pass",
            "score": keep,
            "payout": 0,
        }
        chosen, rejected = (taking, passing) if wants_it else (passing, taking)
        rejected["behind"] = chosen["score"] - rejected["score"]

        return Recommendation(
            seat=request.seat,
            action=chosen["action"],
            action_label=chosen["label"],
            confidence=self._confidence(abs(take - keep)),
            risk_alpha=self.risk_alpha,
            # Options *other than* the recommendation, matching how a discard
            # reports its runners-up. A list that included the chosen action would
            # make "the alternatives" mean two different things by decision type.
            alternatives=[rejected],
            reasoning=self._explain_call(call, keep, take, wants_it, decision),
        )

    # ------------------------------------------------------------ wording ----
    #
    # Kept to plain ASCII. This text is rendered into a browser panel, a
    # transparent overlay and occasionally a terminal, and a dash that survives two
    # of those is worse than one that survives all three.

    # A recommendation this unsure is a toss-up, and naming the consideration that
    # "decided" it would dress up sampling noise as reasoning.
    TOSS_UP = 0.65

    def _explain_discard(self, best, runner_up, confidence: float) -> str:
        parts = [
            f"Throw {best['label']}. What is left is worth about "
            f"{best['hand_value']:.0f} coins, and it risks {best['danger']:.0f} "
            f"if somebody claims it."
        ]
        if runner_up is None:
            return " ".join(parts)

        if confidence < self.TOSS_UP:
            parts.append(
                f"This one is a coin flip: {runner_up['label']} is within "
                f"{best['score'] - runner_up['score']:.0f} coins, which is inside "
                f"the sampling noise. Either is fine."
            )
            return " ".join(parts)

        offence = best["hand_value"] - runner_up["hand_value"]
        defence = runner_up["danger"] - best["danger"]
        # Saying *which* consideration decided it is the part a human can argue
        # with, and arguing with it is how the rules get corrected.
        if defence > abs(offence):
            parts.append(
                f"Mainly a safety call: {runner_up['label']} would risk "
                f"{runner_up['danger']:.0f}, which is {defence:.0f} more."
            )
        else:
            parts.append(
                f"Mainly about what it keeps: throwing {runner_up['label']} instead "
                f"would leave {offence:.0f} coins less on the table."
            )
        return " ".join(parts)

    def _explain_call(self, call, keep, take, wants_it, decision: DecisionType) -> str:
        shape = call.describe(self.agent.space)
        if wants_it:
            if decision is DecisionType.CHAIN:
                return (
                    f"Call again: {shape}. The refill can keep the chain going, "
                    f"though every card it draws shortens the deck."
                )
            return f"Take it: {shape}. That beats holding on, worth about {keep:.0f}."
        return (
            f"Pass. {shape} pays {call.payout}, but the hand it breaks up is worth "
            f"about {keep:.0f}, more than the {take:.0f} you would be left with."
        )

    # --------------------------------------------------------- confidence ----
    def _confidence(self, gap: float) -> float:
        """How likely the ranking is to survive redrawing the belief.

        Grounded in a measurement rather than invented: the danger estimate moves
        by a known amount between independent draws, so a margin can be turned into
        a probability instead of a vibe.

        It answers exactly one question — would resampling change the answer — and
        deliberately not the bigger one, which is whether the heuristic's model of
        the game is right. Model error dominates, so treat this as an upper bound.
        """
        particles = max(1, self.agent.particles)
        sd = DANGER_SD_AT_48 * (BASELINE_PARTICLES / particles) ** 0.5
        if sd <= 0.0:
            return 1.0
        return 0.5 * (1.0 + math.erf(gap / (2.0 * sd)))
