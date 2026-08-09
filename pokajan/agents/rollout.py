"""The cheap policy that plays out determinized worlds.

This exists because of a measurement. `HeuristicAgent.hand_value` costs ~500us a
call: it prices all 84 targets a roster admits, and a discard decision prices them
once per candidate card. That is fine for a policy asked a few dozen times a game,
and hopeless for one asked a hundred thousand times a second inside a search. A
PIMC decision is on the order of 40 rollouts of 25 moves; at heuristic speed that
is 30 seconds per decision.

So the rollout policy is a different animal, built to the opposite priority. Two
things make it fast:

**It only prices the targets a card can affect.** Discarding a blue AzKi cannot
change what a Gawr Gura triple is worth. Only targets involving that one character
move, which is about eight of the eighty-four, and only those get recomputed.

**It replaces probability with a discount per missing card.** No belief, no
availability, no exponentiation — a table lookup on how many cards short a hand
is. Two tables, because needing *any colour* of a character and needing one
*specific* card are different problems: the first is roughly a coin flip over a
game, the second closer to one in four.

It is perfect-information by construction: inside a determinized world the hands
are known, so discard danger is computed exactly rather than sampled. That is
standard for PIMC and it is also the source of PIMC's well-known blind spot —
every simulated opponent plays as though it can see through the table, so the
search systematically underrates hiding information. Worth remembering before
reading too much into any single line it produces.

Numbers here are policy, not rules.
"""

from __future__ import annotations

import random

from ..core.actions import DecisionType, call_action, pass_action
from ..core.cards import Counts
from ..core.rules import HandKind, Rules
from ..core.state import GameState
from ..envs.belief import Belief
from ..server.protocol import ActionResponse, DecisionRequest, PublicState

# Chance-of-completion stand-ins, per card still missing. Needing any colour of a
# character is far more likely than needing one exact card, and collapsing the two
# into a single discount is what makes cheap policies misvalue monochrome hands.
MIXED_STEP = 0.55
MONO_STEP = 0.25

# Weight on expected coins billed if a discard is claimed. 1.0 because both sides
# are already denominated in coins — the same reasoning as the full heuristic.
DANGER_WEIGHT = 1.0


class RolloutPolicy:
    """Plays a known world to the end, quickly."""

    def __init__(self, rules: Rules, *, seed: int | None = None) -> None:
        if rules.bonus_applies_to != "scoring_set":
            # Loud rather than approximate. The fast path folds the bonus into a
            # precomputed payout per target, which is only correct when the bonus
            # counts the scoring cards. Under `whole_hand` it depends on the rest
            # of the hand and cannot be precomputed — and a search quietly using
            # the wrong payouts would look like a weak agent, not like a bug.
            raise ValueError(
                "RolloutPolicy needs payouts.bonus.applies_to == 'scoring_set'; "
                f"got {rules.bonus_applies_to!r}"
            )
        self.rules = rules
        self.space = rules.cards
        self.rng = random.Random(seed)
        self._call = call_action(rules)
        self._plans: dict[int | None, list[list[tuple]]] = {}

    # ------------------------------------------------------------ planning --
    def _character_targets(self, bonus: int | None) -> list[list[tuple]]:
        """Per character, the targets that use it — fully resolved ahead of time.

        Each entry is `(kind, needed, others, colour, payout)` where `needed` and
        `others` are already the exact slot or member tuples the inner loops walk.
        Everything that can be hoisted out of a rollout is hoisted here: payouts
        with the bonus folded in, group membership minus the character itself, and
        the per-colour slot indices. What remains in the hot path is integer
        comparison.

            kind 0  mixed triple      needed unused
            kind 1  mono triple       needed = the one slot, three copies of it
            kind 2  mixed group       needed = members, others = members minus this one
            kind 3  mono group        needed = member slots in colour, others likewise
        """
        cached = self._plans.get(bonus)
        if cached is not None:
            return cached

        rules, space = self.rules, self.space
        per_copy = rules.bonus_per_copy

        plans: list[list[tuple]] = []
        for c in range(space.n_chars):
            entries: list[tuple] = []
            triple_bonus = 3 * per_copy if c == bonus else 0
            entries.append((0, None, None, None, rules.payout(HandKind.TRIPLE) + triple_bonus))
            mono_triple = rules.payout(HandKind.TRIPLE, monochrome=True) + triple_bonus
            for k in range(space.n_colors):
                entries.append((1, space.char_slots[c][k], None, k, mono_triple))

            for g in space.char_groups[c]:
                members = space.group_members[g]
                size = len(members)
                group_bonus = per_copy if bonus in members else 0
                others = tuple(m for m in members if m != c)
                entries.append(
                    (
                        2,
                        members,
                        others,
                        None,
                        rules.payout(HandKind.GROUP, group_size=size) + group_bonus,
                    )
                )
                mono_group = (
                    rules.payout(HandKind.GROUP, group_size=size, monochrome=True)
                    + group_bonus
                )
                for k in range(space.n_colors):
                    entries.append(
                        (
                            3,
                            tuple(space.char_slots[m][k] for m in members),
                            tuple(space.char_slots[m][k] for m in others),
                            k,
                            mono_group,
                        )
                    )
            plans.append(entries)

        self._plans[bonus] = plans
        return plans

    # -------------------------------------------------------------- acting --
    def choose(
        self,
        state: GameState,
        seat: int,
        decision: DecisionType,
        claimable_slot: int | None = None,
    ) -> int:
        """The action this seat plays in a rollout."""
        if decision is DecisionType.DISCARD:
            return self._discard(state, seat)
        # Calling, claiming and chaining are taken greedily. Whether to decline is
        # a genuinely hard question — a chain drains the shared deck and can end
        # the game — but it is one the search answers at the root, where it can
        # afford to. Inside a rollout it is not worth the cycles.
        return self._call

    def _discard(self, state: GameState, seat: int) -> int:
        space = self.space
        hand = state.hands[seat]
        candidates = [slot for slot in range(space.n_slots) if hand[slot] > 0]
        if not candidates:
            return 0
        if len(candidates) == 1:
            return candidates[0]

        plans = self._character_targets(state.bonus_character)
        totals = space.char_totals(hand)
        danger = self._danger(state, seat, candidates, plans)
        return self.best_discard(plans, hand, totals, candidates, danger)

    def best_discard(self, plans, hand: Counts, totals, candidates, danger) -> int:
        """The card whose loss costs least, net of what it would be billed if claimed.

        Only targets involving the discarded card's character can change, so the
        comparison is between one character's value before and after — never a
        rescan of the whole roster. `hand` and `totals` are mutated and restored in
        place rather than copied, which matters at rollout frequency.
        """
        slot_char = self.space.slot_char
        best, best_score = candidates[0], -1e18
        cached: dict[int, float] = {}

        for slot in candidates:
            character = slot_char[slot]
            before = cached.get(character)
            if before is None:
                before = self._character_value(plans, hand, totals, character)
                cached[character] = before

            hand[slot] -= 1
            totals[character] -= 1
            after = self._character_value(plans, hand, totals, character)
            hand[slot] += 1
            totals[character] += 1

            score = -(before - after) - DANGER_WEIGHT * danger[slot]
            if score > best_score:
                best, best_score = slot, score
        return best

    def _character_value(
        self, plans, hand: Counts, totals: list[int], character: int
    ) -> float:
        """Best discounted payout among the targets that use `character`."""
        best = 0.0
        for kind, needed, _others, _color, payout in plans[character]:
            if kind == 0:
                missing = 3 - totals[character]
            elif kind == 1:
                missing = 3 - hand[needed]
            elif kind == 2:
                missing = 0
                for m in needed:
                    if totals[m] == 0:
                        missing += 1
            else:
                missing = 0
                for s in needed:
                    if hand[s] == 0:
                        missing += 1

            if missing <= 0:
                value = float(payout)
            else:
                step = MIXED_STEP if kind == 0 or kind == 2 else MONO_STEP
                value = payout * step**missing
            if value > best:
                best = value
        return best

    # ------------------------------------------------------------- defence --
    def _danger(self, state: GameState, seat: int, candidates: list[int], plans):
        """Exactly what each discard would cost, in this world.

        No sampling: the hands are known here, so this is the real bill rather than
        an estimate — and it is a maximum across opponents, since only one claim
        can win.

        Most candidates cannot be claimed by most opponents, so the cheap
        eligibility test runs first and the payout scan only happens on the rare
        card that is genuinely live. That ordering is most of why this is fast
        enough to sit inside a rollout.
        """
        space = self.space
        slot_char = space.slot_char
        out = {slot: 0.0 for slot in candidates}
        for other in range(state.players):
            if other == seat:
                continue
            hand = state.hands[other]
            totals = space.char_totals(hand)
            for slot in candidates:
                if not _could_claim(plans, totals, slot_char[slot]):
                    continue
                payout = claim_payout(self.rules, plans, hand, totals, slot)
                if payout > out[slot]:
                    out[slot] = payout
        return out


def _could_claim(plans, totals: list[int], character: int) -> bool:
    """The `can_call_using` test, phrased in terms of the precomputed plans.

    Adding one copy of `character` must be what completes the hand: either it makes
    a third copy, or it fills the last hole in a group. Anything else already scores
    without the card, and is a hand the seat must wait to call on its own turn.
    """
    if totals[character] >= 2:
        return True
    for kind, _needed, others, _color, _payout in plans[character]:
        if kind != 2:
            continue
        for m in others:
            if totals[m] == 0:
                break
        else:
            return True
    return False


def claim_payout(rules: Rules, plans, hand: Counts, totals: list[int], slot: int) -> float:
    """What `hand` could collect by claiming `slot`, or 0 if it could not.

    Mirrors `evaluate.best_call(..., must_use=slot)` without building any objects:
    only hands that actually *spend* the claimed card count, which is the rule that
    stops a made hand from claiming anything it likes. Kept honest by a property
    test that runs it against `best_call` on random hands and configs.

    Assumes the bonus counts the scoring cards, which `RolloutPolicy` enforces at
    construction — the payouts in `plans` already have it folded in.
    """
    space = rules.cards
    character = space.slot_char[slot]
    color = space.slot_color[slot]
    best = 0.0

    for kind, _needed, others, want_color, payout in plans[character]:
        if payout <= best:
            continue
        if kind == 0:
            ok = totals[character] >= 2
        elif kind == 1:
            ok = want_color == color and hand[slot] >= 2
        elif kind == 2:
            ok = True
            for m in others:
                if totals[m] == 0:
                    ok = False
                    break
        else:
            if want_color != color:
                continue
            ok = True
            for s in others:
                if hand[s] == 0:
                    ok = False
                    break

        if ok:
            best = float(payout)
    return best


class FastAgent:
    """The rollout policy's valuation, with the belief supplying what it cannot see.

    `RolloutPolicy` is strong — with perfect information it beats the full
    heuristic by ~520 coins/game — and roughly 275x cheaper per decision. Almost
    all of that cheapness survives the move to imperfect information: the only part
    that genuinely needs hidden cards is discard danger, and belief particles
    supply it.

    The result sits between `GreedyCallerAgent` and `HeuristicAgent`: far faster
    than the heuristic and far stronger than greedy. That combination is what a
    self-play training loop wants at M5, where the opponent pool is evaluated
    millions of times and 29ms a decision is not a budget anyone has.
    """

    def __init__(
        self,
        rules: Rules,
        *,
        seed: int | None = None,
        particles: int = 8,
        defend: bool = True,
        name: str = "fast",
    ) -> None:
        self.rules = rules
        self.space = rules.cards
        self.policy = RolloutPolicy(rules, seed=seed)
        self.particles = particles
        self.defend = defend
        self.name = name
        self._seed = seed
        self._call = call_action(rules)
        self._pass = pass_action(rules)

        self.belief: Belief | None = None
        self._game_id: str | None = None
        self._bonus: int | None = None

    def act(self, request: DecisionRequest) -> ActionResponse:
        state = request.state
        if self._game_id != state.game_id or self.belief is None:
            self._game_id = state.game_id
            self._bonus = state.bonus_character
            self.belief = Belief(
                self.rules, state.viewer, seed=self._seed, bonus_character=self._bonus
            )
        self.belief.observe(state)

        if DecisionType[request.decision] is DecisionType.DISCARD:
            action = self._discard(state)
        else:
            # Greedy on calls, claims and chains. Search measurably adds nothing
            # here — see the M4 note in the README — and the heuristic's own
            # comparison almost always says call anyway.
            action = self._call if request.legal_mask[self._call] else self._pass

        if not request.legal_mask[action]:
            action = next(i for i, ok in enumerate(request.legal_mask) if ok)
        return ActionResponse(game_id=request.game_id, seat=request.seat, action=action)

    def _discard(self, state: PublicState) -> int:
        space = self.space
        hand = list(state.hand)
        candidates = [slot for slot in range(space.n_slots) if hand[slot] > 0]
        if not candidates:
            return 0
        if len(candidates) == 1:
            return candidates[0]

        plans = self.policy._character_targets(self._bonus)
        totals = space.char_totals(hand)
        danger = self._danger(state, candidates, plans)
        return self.policy.best_discard(plans, hand, totals, candidates, danger)

    def _danger(self, state: PublicState, candidates: list[int], plans):
        """Expected coins billed if this discard is claimed, averaged over worlds."""
        out = {slot: 0.0 for slot in candidates}
        if not self.defend or self.particles <= 0:
            return out

        space = self.space
        slot_char = space.slot_char
        viewer = state.viewer

        for particle in self.belief.sample(self.particles):
            # Within one world, the biggest payout wins the card — so this is a
            # maximum across opponents. Across worlds it is an expectation, so the
            # per-world worst case is what gets weighted in.
            worst = {slot: 0.0 for slot in candidates}
            for seat in range(len(state.seats)):
                if seat == viewer:
                    continue
                hand = particle.hands[seat]
                totals = space.char_totals(hand)
                for slot in candidates:
                    if not _could_claim(plans, totals, slot_char[slot]):
                        continue
                    payout = claim_payout(self.rules, plans, hand, totals, slot)
                    if payout > worst[slot]:
                        worst[slot] = payout
            for slot in candidates:
                out[slot] += particle.weight * worst[slot]
        return out


def play_out(
    engine,
    policy: RolloutPolicy,
    *,
    max_rounds: int = 400,
) -> list[int]:
    """Run a determinized game to the end and return the final coins.

    Driven through `pending_seats`/`apply` rather than the client-facing
    `pending_decisions`/`submit`: building a `DecisionRequest` per seat copies
    every count vector in the game, which measured 79x more expensive than asking
    the same question directly, and a rollout never reads one.
    """
    state = engine.state
    rounds = 0
    while not state.finished and rounds < max_rounds:
        pending = engine.pending_seats()
        if not pending:
            break
        engine.apply(
            [
                (seat, policy.choose(state, seat, decision, slot))
                for seat, decision, slot in pending
            ]
        )
        rounds += 1
    return state.coins
