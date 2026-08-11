"""A hand-written agent good enough to be a real opponent.

Three jobs: give the learned agents something to beat, label decisions for
behaviour cloning at M5, and act as the rollout policy inside PIMC at M4. All
three want the same thing — decent play that is fast and has no hidden state.

The shape of the reasoning follows from one property of this game that is easy to
state and easy to get wrong. **Hand strength is the payout**, and the payouts are
not close together: a monochrome triple pays 840 where a mixed one pays 120, and a
monochrome five-group pays 1800, more than the starting stack. So counting cards
from completion — the natural mahjong instinct — ranks hands wrongly. A hand two
cards from 1800 is worth more than a hand one card from 120. Everything below is
scored in expected coins instead, and shanten only ever enters as an input to a
probability.

The other half is defence, and it matters more here than in most card games: a
claimed hand bills **the discarder alone**. Throwing the wrong card does not cost
you a share of a pot, it costs you the whole thing, and a monochrome five-group
off your discard ends the game with you bankrupt. So a discard is scored as
`what the rest of my hand becomes worth` minus `what I expect to be billed`, both
in coins, which is why the two terms need no relative weighting.

Numbers in this file are policy, not rules. They describe how this agent guesses,
not how the game works, so they stay out of rules/pokajan_v1.yaml.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..core.actions import DecisionType, call_action, pass_action
from ..core.cards import Counts
from ..core.evaluate import Call, best_call
from ..core.rules import HandKind, Rules
from ..envs.belief import Belief
from ..server.protocol import ActionResponse, DecisionRequest, PublicState

# Fraction of opponent turns that actually offer a card you need. Every opponent
# turn produces exactly one claimable discard, but they throw what they do not
# want, which correlates with what nobody wants. Below 1.0 to say so.
CLAIM_EFFICIENCY = 0.55

# Particles drawn per decision. The discard scan reuses them for the danger
# estimate, so this buys both the read and the defence.
DEFAULT_PARTICLES = 48

# Sampled futures per decision when Monte-Carlo valuation is switched on. 0 uses
# the closed form instead.
DEFAULT_FUTURES = 32

# How many of this seat's future draws a sampled future contains.
#
# The obvious choice is "all of them" — deck_remaining / players, around 17 at
# mid-game — and it is wrong, because the sampled hand is never made to obey the
# hand limit. Handed 17 extra cards, a seat can assemble essentially any target,
# so the valuation stops asking "what will this hand become" and starts asking
# "what exists in the deck". Capping the horizon keeps the reachable hand near the
# size a real one can actually hold.
#
# Swept against the closed-form heuristic, coins/game:
#
#     horizon 2    -37   (CI [-88, +14])
#     horizon 5    -19   (CI [-69, +32])   <- default
#     horizon 10   -84   (CI [-135, -34])
#     uncapped    -132   (CI [-170, -94])
#
# So the cap is worth 113 coins over the uncapped version, and the best setting
# still only draws level with the closed form it was meant to replace.
DEFAULT_HORIZON = 5

# Chance that a card passing through an opponent's hand is thrown at a moment it
# would complete your hand — the one place claims enter the sampled valuation.
#
# It replaces CLAIM_EFFICIENCY rather than joining it, and is easier to reason
# about: that constant scaled an abstract count of "opportunities", where this is
# a probability attached to a specific card in a specific simulated deck.
CLAIM_ARRIVAL_RATE = 0.4

# How many targets a hand's value may be built from. 1 means a plain maximum.
#
# Combining the best four was tried, on the reasoning that two live chances are
# worth more than one and a maximum cannot express that. Measured over 480
# duplicate-dealt games it came out at -38 coins/game (95% CI [-89, +13]) — no
# evidence of improvement and a hint of harm. The likely reason is that targets
# overlap heavily: a mixed triple and a monochrome triple of the same character
# compete for the same cards, so treating them as independent chances doubles up
# on hands the seat cannot actually hold at once.
#
# Left parameterised rather than deleted because the finding is about *this*
# completion-probability model, not about the idea. `heuristic-combined` in
# train/evaluate.py re-runs the comparison.
TARGETS_COMBINED = 1

# Scales the defensive term. 1.0 is the principled setting — both sides of the
# comparison are already expected coins — and it is exposed only so the ablation
# in tests/scenarios can turn defence off and show it was worth having.
DANGER_WEIGHT = 1.0


@dataclass(frozen=True)
class Target:
    """One hand this seat could aim at, priced before any cards are held."""

    kind: HandKind
    character: int | None
    group: int | None
    color: int | None            # None means mixed-colour
    size: int | None             # group size
    bonus_copies: int            # copies of the bonus holomem in the scoring set


def build_targets(rules: Rules, bonus_character: int | None) -> list[Target]:
    """Every scoring hand the roster admits.

    Enumerated once per game rather than searched per decision: for a 17-holomem
    roster this is 17 mixed triples, 51 monochrome triples, 4 mixed groups and 12
    monochrome groups — 84 entries, small enough to score exhaustively every time.
    """
    space = rules.cards
    targets: list[Target] = []

    for c in range(space.n_chars):
        bonus = 3 if c == bonus_character else 0
        targets.append(Target(HandKind.TRIPLE, c, None, None, None, bonus))
        for k in range(space.n_colors):
            targets.append(Target(HandKind.TRIPLE, c, None, k, None, bonus))

    for g, members in enumerate(space.group_members):
        bonus = 1 if bonus_character in members else 0
        size = len(members)
        targets.append(Target(HandKind.GROUP, None, g, None, size, bonus))
        for k in range(space.n_colors):
            targets.append(Target(HandKind.GROUP, None, g, k, size, bonus))

    return targets


class HeuristicAgent:
    """Expected-coins play with belief-driven discard safety."""

    def __init__(
        self,
        rules: Rules,
        *,
        seed: int | None = None,
        particles: int = DEFAULT_PARTICLES,
        futures: int = 0,
        horizon: int = DEFAULT_HORIZON,
        defend: bool = True,
        targets_combined: int = TARGETS_COMBINED,
        name: str = "heuristic",
    ) -> None:
        self.rules = rules
        self.space = rules.cards
        self.rng = random.Random(seed)
        self.particles = particles
        # 0 keeps the closed-form valuation; anything higher samples that many
        # futures per decision instead. See `hand_value_sampled`.
        self.futures = futures
        self.horizon = max(1, horizon)
        self.defend = defend
        # 1 reduces hand valuation to a plain maximum over targets; the
        # `heuristic-max` entry in train/evaluate.py uses it to measure what
        # combining chances is worth.
        self.targets_combined = max(1, targets_combined)
        self.name = name
        self._seed = seed

        self.belief: Belief | None = None
        self.targets: list[Target] = []
        self._game_id: str | None = None
        self._bonus: int | None = None

    # ------------------------------------------------------------- acting ---
    def act(self, request: DecisionRequest) -> ActionResponse:
        state = request.state
        self.start_game(state)
        self.belief.observe(state)

        decision = DecisionType[request.decision]
        if decision is DecisionType.DISCARD:
            action = self.act_discard(state)
        elif decision is DecisionType.CLAIM:
            action = self._choose_claim(state, request.claimable_slot)
        else:
            action = self._choose_call(state)

        if not request.legal_mask[action]:
            # Never expected, but a heuristic that quietly proposes an illegal move
            # would surface as a mysterious engine error rather than as itself.
            legal = [i for i, ok in enumerate(request.legal_mask) if ok]
            action = legal[0]

        return ActionResponse(game_id=request.game_id, seat=request.seat, action=action)

    def start_game(self, state: PublicState) -> None:
        if self._game_id == state.game_id and self.belief is not None:
            return
        self._game_id = state.game_id
        self._bonus = state.bonus_character
        self.belief = Belief(
            self.rules, state.viewer, seed=self._seed, bonus_character=self._bonus
        )
        self.targets = build_targets(self.rules, self._bonus)

    # ----------------------------------------------------------- decisions --
    def act_discard(self, state: PublicState, particles=None) -> int:
        """Which card to throw. Public so search can defer to it.

        `particles` lets a caller that has already sampled the belief — PIMC does,
        for its determinizations — reuse them instead of paying for a second draw
        and then disagreeing with itself about the world.
        """
        hand = state.hand
        candidates = [s for s in range(self.space.n_slots) if hand[s] > 0]
        if not candidates:
            return 0

        if particles is None:
            particles = self._particles()
        elif not self.defend:
            particles = []
        value = self.valuer(state, particles)
        danger = self.danger(candidates, particles, state)

        best, best_score = candidates[0], -float("inf")
        for slot in candidates:
            trial = hand[:]
            trial[slot] -= 1
            score = value(trial) - DANGER_WEIGHT * danger.get(slot, 0.0)
            if score > best_score:
                best, best_score = slot, score
        return best

    def _choose_claim(self, state: PublicState, slot: int | None) -> int:
        """Take the card, unless taking it destroys something worth more."""
        if slot is None:
            return pass_action(self.rules)

        probe = state.hand[:]
        probe[slot] += 1
        call = best_call(
            self.rules, probe, bonus_character=self._bonus, claimed=True, must_use=slot
        )
        if call is None:
            return pass_action(self.rules)

        value = self.valuer(state, self._particles())
        # Declining leaves the hand exactly as it is; claiming banks the payout and
        # leaves the remnant. Both sides are in coins.
        keep = value(state.hand)
        take = call.payout + value(_spend(probe, call))
        return call_action(self.rules) if take >= keep else pass_action(self.rules)

    def _choose_call(self, state: PublicState) -> int:
        call = best_call(self.rules, state.hand, bonus_character=self._bonus)
        if call is None:
            return pass_action(self.rules)

        value = self.valuer(state, self._particles())
        keep = value(state.hand)
        take = call.payout + value(_spend(state.hand, call))
        return call_action(self.rules) if take >= keep else pass_action(self.rules)

    # ------------------------------------------------------------ scoring ---
    def _particles(self):
        if self.particles <= 0 or (not self.defend and self.futures <= 0):
            return []
        return self.belief.sample(self.particles)

    def valuer(self, state: PublicState, particles):
        """A function from hand to expected coins, built once per decision.

        Both valuations are expensive to set up and cheap to reuse — the closed
        form needs the belief's availability vector, the sampled one needs a set of
        drawn futures — so the setup happens here and every candidate discard is
        scored against the same one. Comparing candidates against *different*
        samples would put the noise between them rather than around them.
        """
        if self.futures <= 0:
            context = self.valuation_context(state)
            return lambda hand: self.hand_value(hand, context)
        futures = self.sample_futures(state, particles)
        return lambda hand: self.hand_value_sampled(hand, futures)

    def sample_futures(self, state: PublicState, particles):
        """Which cards this seat will draw, and which it might get to claim.

        This is the alternative to searching that M4 argued for. A full rollout
        answers "how does the game end", which turns out to be chaotic — one
        different discard flips a claim, a claim changes a refill, and every
        later draw moves. This answers only "which cards reach me", which is
        smooth: it depends on the deck and on turn order, not on a cascade of
        decisions. So it absorbs sampling effort productively where the rollout
        did not.

        Claims enter once, and only in the way the rules allow: a claim must
        *complete* a hand, so a claimable card can finish a target that is one
        short and can never advance one that is two short.
        """
        space = self.space
        players = self.rules.play.players
        if not particles:
            particles = self.belief.sample(max(1, min(self.futures, self.particles or 8)))

        mine = max(0, min(self.horizon, state.deck_remaining // players))
        futures = []
        for i in range(self.futures):
            particle = particles[i % len(particles)]
            deck = [slot for slot, n in enumerate(particle.deck) for _ in range(n)]
            self.rng.shuffle(deck)

            drawn = space.zeros()
            for slot in deck[:mine]:
                drawn[slot] += 1
            claimable = {
                slot for slot in deck[mine:] if self.rng.random() < CLAIM_ARRIVAL_RATE
            }
            futures.append((drawn, claimable))
        return futures

    def hand_value_sampled(self, hand: Counts, futures) -> float:
        """Expected coins, as the mean best payout actually reachable.

        Note what this measures compared with the closed form: not
        `max over targets of payout x P(complete)`, but `E[best payout that
        completes]`. The difference is that a hand with two live chances is worth
        more than either alone, and this expresses that without the
        double-counting that made combining targets fail at M3 — each sampled
        future contributes exactly one payout, the best one it actually reaches.
        """
        n = len(hand)
        total = 0.0
        for drawn, claimable in futures:
            reachable = [hand[i] + drawn[i] for i in range(n)]
            best = 0
            for target in self.targets:
                missing = self._requirements(reachable, target)
                if missing is None:
                    continue
                if missing:
                    # One short is claimable; two short is not reachable at all.
                    if len(missing) != 1:
                        continue
                    if not any(slot in claimable for slot in missing[0]):
                        continue
                payout = self._payout(reachable, target)
                if payout > best:
                    best = payout
            total += best
        return total / len(futures)

    def valuation_context(self, state: PublicState) -> tuple[list[float], float, float, float]:
        """Everything hand valuation needs about the wider game, computed once."""
        unseen = self.belief.expected_unseen()
        total = sum(unseen)
        players = self.rules.play.players
        my_draws = state.deck_remaining / players
        claim_ops = state.deck_remaining * (players - 1) / players * CLAIM_EFFICIENCY
        return unseen, max(total, 1e-9), my_draws, claim_ops

    def hand_value(self, hand: Counts, context) -> float:
        """Expected coins this hand is on track to earn.

        Not a sum over targets: only one hand may score at a time and a card may
        only be used once, so two chases are not worth twice one. By default it is
        a plain maximum, which has a known blind spot — an agent is indifferent to
        throwing away its second-best chance, since doing so leaves the best one
        untouched. Combining the top few instead is implemented here and switched
        off, because measuring it showed it does not help; see TARGETS_COMBINED.

        Each target is taken only in the worlds where nothing better landed, so a
        payout is never double-counted however many are combined.
        """
        candidates = []
        for target in self.targets:
            reqs = self._requirements(hand, target)
            if reqs is None:
                continue
            payout = self._payout(hand, target)
            probability = 1.0 if not reqs else completion_probability(reqs, *context)
            if probability > 0.0:
                candidates.append((payout * probability, probability, payout))

        if not candidates:
            return 0.0

        candidates.sort(reverse=True)
        value, unclaimed = 0.0, 1.0
        for _, probability, payout in candidates[: self.targets_combined]:
            value += unclaimed * probability * payout
            unclaimed *= 1.0 - probability
            if unclaimed <= 0.0:
                break
        return value

    def _payout(self, hand: Counts, target: Target) -> int:
        copies = target.bonus_copies
        if self.rules.bonus_applies_to == "whole_hand" and self._bonus is not None:
            copies = sum(hand[s] for s in self.space.char_slots[self._bonus])
        return self.rules.payout(
            target.kind,
            group_size=target.size,
            monochrome=target.color is not None,
            bonus_copies=copies,
        )

    def _requirements(self, hand: Counts, target: Target) -> list[tuple[int, ...]] | None:
        """Acceptable slots for each card still needed, or None if unreachable.

        A monochrome triple, for instance, needs three copies of one exact slot, so
        each requirement admits exactly one slot; a mixed triple admits any colour
        of that character. That distinction is the whole reason hands are valued
        against the belief rather than against a raw count of unseen cards.
        """
        space = self.space
        if target.kind is HandKind.TRIPLE:
            if target.color is None:
                held = space.char_totals(hand)[target.character]
                return [space.char_slots[target.character]] * max(0, 3 - held)
            slot = space.char_slots[target.character][target.color]
            if 3 > space.max_per_color:
                return None
            return [(slot,)] * max(0, 3 - hand[slot])

        members = space.group_members[target.group]
        if target.color is None:
            totals = space.char_totals(hand)
            return [space.char_slots[m] for m in members if totals[m] == 0]
        return [
            (space.char_slots[m][target.color],)
            for m in members
            if hand[space.char_slots[m][target.color]] == 0
        ]

    # ------------------------------------------------------------ defence ---
    def danger(self, candidates: list[int], particles, state: PublicState) -> dict[int, float]:
        """Expected coins billed to this seat for each discard it might make.

        Averaged over belief particles, and a maximum rather than a sum across
        opponents because only one claim can win. The particles already carry the
        pass evidence, so a card the table has declined once scores as safe without
        that having to be special-cased here.
        """
        if not particles:
            return {}

        space = self.space
        rules = self.rules
        viewer = state.viewer
        opponents = [v.seat for v in state.seats if v.seat != viewer]
        out = {slot: 0.0 for slot in candidates}

        for particle in particles:
            worst = {slot: 0 for slot in candidates}
            for seat in opponents:
                hand = particle.hands[seat]
                totals = space.char_totals(hand)
                for slot in candidates:
                    character = space.slot_char[slot]
                    # Inlined eligibility: adding this card must be what completes
                    # the hand. Hoisting char_totals out of the slot loop is what
                    # keeps a 48-particle scan affordable.
                    if totals[character] + 1 < 3 and not any(
                        all(totals[m] >= 1 for m in space.group_members[g] if m != character)
                        for g in space.char_groups[character]
                    ):
                        continue
                    probe = hand[:]
                    probe[slot] += 1
                    call = best_call(
                        rules, probe, bonus_character=self._bonus, claimed=True, must_use=slot
                    )
                    if call is not None and call.payout > worst[slot]:
                        worst[slot] = call.payout
            for slot in candidates:
                out[slot] += particle.weight * worst[slot]
        return out


def _spend(hand: Counts, call: Call) -> Counts:
    remainder = hand[:]
    for slot, n in enumerate(call.cards):
        remainder[slot] -= n
    return remainder


def completion_probability(
    reqs: list[tuple[int, ...]],
    unseen: list[float],
    unseen_total: float,
    my_draws: float,
    claim_ops: float,
) -> float:
    """Rough chance of collecting every missing card before the game ends.

    Each requirement is treated as independent, which is optimistic — draws that
    fill one do not fill another — but the error is small next to the spread in
    payouts this is being used to compare.

    The one asymmetry worth modelling exactly is that **claims only ever finish a
    hand**. You cannot claim your way from two cards short to one, because a claim
    is legal only when the claimed card completes the hand. So exactly one
    requirement gets the extra opportunities, and it is given to the scarcest —
    the one most likely to still be outstanding at the end.
    """
    if not reqs:
        return 1.0

    rates = []
    for slots in reqs:
        available = sum(unseen[s] for s in slots)
        if available <= 0.0:
            return 0.0
        rates.append(available / unseen_total)

    scarcest = min(range(len(rates)), key=lambda i: rates[i])
    probability = 1.0
    for i, rate in enumerate(rates):
        opportunities = my_draws + (claim_ops if i == scarcest else 0.0)
        probability *= 1.0 - (1.0 - rate) ** opportunities
    return probability
