"""What is in the deck, and where it is.

This is the module the whole project's edge rests on, so it is worth being
explicit about why it exists at all.

The real game shows a "remaining cards" list. It is a decoy. It counts every card
the player has not *seen*, drawn from the full theoretical pool of 9 copies per
holomem — 153 cards for a 17-holomem lineup — while the deck only ever holds 100.
About a third of the entries are cards that are not in the game. A human reading
that list is being confidently misled and has no practical way not to be, because
the composition rule is never revealed.

So inference here is two-level, and the first level is the one humans cannot do:

  1. **Composition.** Which 100 of the possible cards were dealt into this game at
     all. Latent, never observed, only ever inferred from cards that have actually
     surfaced.
  2. **Location.** Given a composition, where the not-yet-seen cards sit — split
     between three opponent hands and the draw pile.

Level 1 is solved analytically per slot; level 2 by weighted particles, because
the sharpest evidence it has is a *joint* constraint over several slots at once
(see the pass signal below) and a per-slot mean field would blunt it to nothing.
Particles are also exactly what PIMC wants at M4, so the two consumers share one
implementation rather than drifting apart.

**Numbers in this file are not rules.** The evidence weights and decay below are
policy parameters — how much to trust an inference — not statements about how the
game works. They stay here rather than in rules/pokajan_v1.yaml on purpose: that
file is the description of the game, and nothing in it should change because our
inference got better.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..core.cards import Counts
from ..core.evaluate import can_call_using
from ..core.rules import Rules
from ..core.state import build_deck
from ..server.protocol import PublicState

# --------------------------------------------------------------------------
# Inference parameters. Tunable policy, not rules.
# --------------------------------------------------------------------------

# How much prior mass to move off the configured composition rule and onto a flat
# distribution. The rule is a genuine unknown, so a prior that called the real
# game's composition *impossible* would leave the posterior permanently unable to
# recover from it — the exact failure mode the in-game counter has. This is the
# knob that keeps the model honest about not knowing.
DEFAULT_UNCERTAINTY = 0.15

# Weight multiplier for a particle in which a seat holds cards that would have let
# it claim a discard it demonstrably passed on. Small, not zero: the engine only
# asks seats that *could* claim, so a card left on the table is strong evidence of
# ineligibility but not proof — an eligible seat is allowed to decline.
PASS_VIOLATION = 0.04

# Likewise for a seat holding a card it recently threw away. Larger, because
# discarding one copy while holding another is uncommon but entirely legal.
DISCARD_VIOLATION = 0.35

# Per-turn decay applied to both. A pass tells you about the hand as it was *then*;
# several draws later it says almost nothing, so old evidence fades toward 1.0
# rather than being trusted forever.
EVIDENCE_DECAY = 0.75

# Passes older than this are dropped entirely — they cost time and carry no signal.
PASS_MEMORY_TURNS = 8

DEFAULT_PARTICLES = 64
DEFAULT_PRIOR_SAMPLES = 2000


# --------------------------------------------------------------------------
# Level 1 prior: what compositions the rules produce
# --------------------------------------------------------------------------

_PRIOR_CACHE: dict[tuple[str, int], tuple[tuple[float, ...], ...]] = {}


def composition_prior(
    rules: Rules, *, samples: int = DEFAULT_PRIOR_SAMPLES, seed: int = 12345
) -> tuple[tuple[float, ...], ...]:
    """Per-slot prior over how many copies the deck holds, as `p[slot][count]`.

    Estimated by sampling `build_deck` rather than deriving it in closed form, so
    that a corrected composition rule needs no matching edit here. Cached per
    rules hash — this is called once per game at most.
    """
    key = (rules.rules_hash, samples)
    cached = _PRIOR_CACHE.get(key)
    if cached is not None:
        return cached

    rng = random.Random(seed)
    n_slots = rules.cards.n_slots
    cap = rules.cards.max_per_color
    tally = [[0] * (cap + 1) for _ in range(n_slots)]

    for _ in range(samples):
        counts = [0] * n_slots
        for slot in build_deck(rules, rng):
            counts[slot] += 1
        for slot, k in enumerate(counts):
            tally[slot][k] += 1

    prior = tuple(tuple(t / samples for t in row) for row in tally)
    _PRIOR_CACHE[key] = prior
    return prior


def _blend(prior: tuple[tuple[float, ...], ...], uncertainty: float) -> list[list[float]]:
    """Mix the sampled prior toward flat, so no count is ruled out a priori."""
    out = []
    for row in prior:
        flat = 1.0 / len(row)
        out.append([(1.0 - uncertainty) * p + uncertainty * flat for p in row])
    return out


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n or n < 0:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _tilt(rows: list[list[float]], target: int) -> list[list[float]]:
    """Reweight independent per-slot distributions so their means total `target`.

    Treating slots independently is what makes the update tractable, but it throws
    away the one thing about composition that is known for certain: the deck holds
    exactly 100 cards, whatever roster it was built from. Applying an exponential
    tilt `p(k) -> p(k) * theta^k` and solving for the theta that restores the total
    is the maximum-entropy way to put that back — it is the least-committal change
    to the marginals that satisfies the constraint.

    It is not bookkeeping. It is where "I have seen four copies of this character,
    so something else must be thinner than I thought" enters the model, and that
    inference is unavailable to a player reading the in-game counter.
    """
    # Bisect on log-theta for the mean, then build the rows once at the answer.
    # Searching and constructing in the same loop is the obvious way to write this
    # and costs ~40x more: it allocates two lists per slot per iteration, when all
    # the search needs is a single scalar.
    width = len(rows[0])
    lo, hi = -30.0, 30.0
    for _ in range(40):
        mid = (lo + hi) * 0.5
        theta = math.exp(mid)
        powers = [1.0] * width
        for k in range(1, width):
            powers[k] = powers[k - 1] * theta

        total = 0.0
        for row in rows:
            mass = 0.0
            weighted_sum = 0.0
            for k in range(width):
                w = row[k] * powers[k]
                mass += w
                weighted_sum += k * w
            if mass > 0.0:
                total += weighted_sum / mass

        if abs(total - target) < 1e-9:
            break
        if total < target:
            lo = mid
        else:
            hi = mid

    theta = math.exp((lo + hi) * 0.5)
    powers = [1.0] * width
    for k in range(1, width):
        powers[k] = powers[k - 1] * theta

    out = []
    for row in rows:
        weighted = [row[k] * powers[k] for k in range(width)]
        mass = sum(weighted)
        out.append(list(row) if mass <= 0.0 else [w / mass for w in weighted])
    return out


# --------------------------------------------------------------------------
# Particles
# --------------------------------------------------------------------------


@dataclass
class Particle:
    """One complete hypothesis about the hidden state.

    `hands` is indexed by absolute seat; the viewer's own entry is their real hand,
    which makes a particle directly usable as a determinized `GameState` at M4.
    """

    hands: list[Counts]
    deck: Counts
    composition: Counts
    weight: float = 1.0


@dataclass
class PassEvent:
    """A discard that went unclaimed — and therefore what nobody could use."""

    turn: int
    slot: int
    discarder: int


# --------------------------------------------------------------------------
# The belief
# --------------------------------------------------------------------------


class Belief:
    """One seat's evolving picture of the hidden state.

    Fed a stream of `PublicState` snapshots via `observe`. Snapshots may skip turns
    — an agent only sees the game when it is asked to act — so evidence is
    recovered by *diffing* consecutive views rather than assuming every event was
    witnessed. That is deliberate: it is exactly the information a screen reader
    will have at M8, watching a real game between its own decisions.
    """

    def __init__(
        self,
        rules: Rules,
        viewer: int,
        *,
        seed: int | None = None,
        bonus_character: int | None = None,
        uncertainty: float = DEFAULT_UNCERTAINTY,
        prior_samples: int = DEFAULT_PRIOR_SAMPLES,
    ) -> None:
        self.rules = rules
        self.viewer = viewer
        self._bonus = bonus_character
        self.rng = random.Random(seed)
        self.space = rules.cards
        self.n_slots = self.space.n_slots
        self.cap = self.space.max_per_color
        self.deck_size = rules.deck_size

        self.prior = _blend(composition_prior(rules, samples=prior_samples), uncertainty)

        self.state: PublicState | None = None
        self.seen: Counts = [0] * self.n_slots
        self.pass_events: list[PassEvent] = []

        self._prev_discards: list[Counts] | None = None
        self._prev_table: Counts | None = None
        self._posterior: list[list[float]] | None = None

    # ---------------------------------------------------------- observing ---
    def observe(self, state: PublicState) -> None:
        """Fold in a new public view."""
        if state.viewer != self.viewer:
            raise ValueError(
                f"belief for seat {self.viewer} was given seat {state.viewer}'s view"
            )

        self._collect_passes(state)

        # Everything publicly accounted for. The union of the viewer's hand, the
        # table and the scored pile is precisely the set of cards drawn from the
        # deck that this seat has seen; the rest is hidden in hands or undrawn.
        self.seen = [
            state.hand[s] + state.table[s] + state.scored[s] for s in range(self.n_slots)
        ]
        self.state = state
        self._posterior = None

        cutoff = state.turn_index - PASS_MEMORY_TURNS
        if self.pass_events and self.pass_events[0].turn < cutoff:
            self.pass_events = [e for e in self.pass_events if e.turn >= cutoff]

    def _collect_passes(self, state: PublicState) -> None:
        """Recover which discards went unclaimed since the last look.

        A discard that stayed on the table is one that *every* other seat declined
        or was ineligible for, which is the strongest read the rules hand out for
        free — and one human players almost never track.
        """
        discards = [list(view.discards) for view in state.seats]
        if self._prev_discards is None:
            self._prev_discards, self._prev_table = discards, list(state.table)
            return

        prev_table = self._prev_table or [0] * self.n_slots
        for slot in range(self.n_slots):
            # Net arrivals on the table: discarded in the gap, minus any claimed.
            stayed = state.table[slot] - prev_table[slot]
            if stayed <= 0:
                continue
            for seat in range(len(discards)):
                if stayed <= 0:
                    break
                if discards[seat][slot] > self._prev_discards[seat][slot]:
                    self.pass_events.append(
                        PassEvent(turn=state.turn_index, slot=slot, discarder=seat)
                    )
                    stayed -= 1

        self._prev_discards, self._prev_table = discards, list(state.table)

    # ------------------------------------------------- level 1: composition --
    def composition_posterior(self) -> list[list[float]]:
        """`p[slot][count]` — how many copies of each card this game was dealt.

        Exact per slot, given the prior. The likelihood is hypergeometric: the
        cards this seat has seen are a subset of a uniformly shuffled deck, so
        seeing `m` copies of a slot out of `K` revealed cards, when the deck holds
        `k` of them among `N`, has probability `C(k,m)C(N-k,K-m)/C(N,K)`.

        Slots are then coupled back together by `_tilt`, so the means total exactly
        `deck_size`. The remaining approximation is that the revealed set is
        treated as uniformly random, when in truth opponents keep the cards they
        like — so the tails are mildly optimistic. That costs little next to
        knowing that a third of the in-game counter is fiction.
        """
        if self._posterior is not None:
            return self._posterior

        n = self.deck_size
        observed = sum(self.seen)
        posterior: list[list[float]] = []

        for slot in range(self.n_slots):
            m = self.seen[slot]
            row = []
            for k in range(self.cap + 1):
                p = self.prior[slot][k]
                if p <= 0.0 or k < m:
                    row.append(0.0)
                    continue
                # C(k, m) * C(n - k, observed - m); the C(n, observed) denominator
                # is shared across k and drops out in the normalisation.
                log_like = _log_choose(k, m) + _log_choose(n - k, observed - m)
                row.append(0.0 if log_like == -math.inf else p * math.exp(log_like))
            total = sum(row)
            if total <= 0.0:
                # Only reachable if more copies have been seen than any allowed
                # count — impossible under a correct config, but falling back to a
                # point mass keeps a rules error from crashing a training run.
                row = [1.0 if k == min(m, self.cap) else 0.0 for k in range(self.cap + 1)]
                total = 1.0
            posterior.append([v / total for v in row])

        # Restore the one certainty the independent update drops: the deck holds
        # exactly `deck_size` cards.
        posterior = _tilt(posterior, self.deck_size)

        self._posterior = posterior
        return posterior

    def expected_composition(self) -> list[float]:
        """Expected copies of each slot in the deck this game was built from."""
        return [
            sum(k * p for k, p in enumerate(row)) for row in self.composition_posterior()
        ]

    def composition_variance(self) -> list[float]:
        out = []
        for row in self.composition_posterior():
            mean = sum(k * p for k, p in enumerate(row))
            out.append(sum(p * (k - mean) ** 2 for k, p in enumerate(row)))
        return out

    def expected_unseen(self) -> list[float]:
        """Expected copies of each slot still hidden — in a hand or in the deck.

        This is the honest version of the number the game displays, and it is the
        one every downstream probability should use.
        """
        mean = self.expected_composition()
        return [max(0.0, mean[s] - self.seen[s]) for s in range(self.n_slots)]

    def hidden_total(self) -> int:
        """Cards not visible to this seat: opponents' hands plus the draw pile."""
        state = self._require_state()
        return state.deck_remaining + sum(
            view.hand_size for view in state.seats if view.seat != self.viewer
        )

    # ---------------------------------------------------- level 2: location --
    def sample(self, count: int = DEFAULT_PARTICLES) -> list[Particle]:
        """Weighted hypotheses about opponents' hands and the remaining deck.

        Importance sampling rather than rejection: a particle that contradicts the
        evidence is down-weighted, never discarded. With rejection, a run of
        unlikely-but-real play could reject every draw and hang the agent; here it
        just degrades toward the prior, which is the correct behaviour when your
        reads turn out to be wrong.
        """
        state = self._require_state()
        posterior = self.composition_posterior()
        others = [v.seat for v in state.seats if v.seat != self.viewer]
        sizes = {v.seat: v.hand_size for v in state.seats}

        particles: list[Particle] = []
        for _ in range(count):
            composition = self._sample_composition(posterior)
            pool = [
                slot
                for slot in range(self.n_slots)
                for _ in range(composition[slot] - self.seen[slot])
            ]
            self.rng.shuffle(pool)

            hands = [[0] * self.n_slots for _ in range(len(state.seats))]
            hands[self.viewer] = list(state.hand)
            cursor = 0
            for seat in others:
                for slot in pool[cursor : cursor + sizes[seat]]:
                    hands[seat][slot] += 1
                cursor += sizes[seat]

            deck = [0] * self.n_slots
            for slot in pool[cursor:]:
                deck[slot] += 1

            particles.append(
                Particle(
                    hands=hands,
                    deck=deck,
                    composition=composition,
                    weight=self._weigh(hands, state),
                )
            )

        total = sum(p.weight for p in particles)
        if total > 0.0:
            for p in particles:
                p.weight /= total
        else:
            for p in particles:
                p.weight = 1.0 / len(particles)
        return particles

    def _sample_composition(self, posterior: list[list[float]]) -> Counts:
        """Draw a whole composition, repaired to hold exactly `deck_size` cards.

        The per-slot posteriors are independent, so their sum wanders off the deck
        total; the repair walks it back. This is what restores the between-slot
        correlation the analytic step dropped, and it is why sampling is worth
        doing even when only the marginals are wanted.
        """
        counts = [self._draw_count(posterior[s], self.seen[s]) for s in range(self.n_slots)]
        total = sum(counts)

        while total > self.deck_size:
            options = [s for s in range(self.n_slots) if counts[s] > self.seen[s]]
            if not options:
                break
            counts[self.rng.choice(options)] -= 1
            total -= 1

        while total < self.deck_size:
            options = [s for s in range(self.n_slots) if counts[s] < self.cap]
            if not options:
                break
            counts[self.rng.choice(options)] += 1
            total += 1

        return counts

    def _draw_count(self, row: list[float], floor: int) -> int:
        r = self.rng.random()
        acc = 0.0
        for k, p in enumerate(row):
            acc += p
            if r <= acc:
                return max(k, floor)
        return max(self.cap, floor)

    def _weigh(self, hands: list[Counts], state: PublicState) -> float:
        """How well one hypothesis explains what players actually did."""
        weight = 1.0
        rules = self.rules
        bonus = self._bonus_character()

        for event in self.pass_events:
            recency = EVIDENCE_DECAY ** max(0, state.turn_index - event.turn)
            if recency < 0.01:
                continue
            for seat in range(len(hands)):
                if seat == self.viewer or seat == event.discarder:
                    continue
                probe = hands[seat][:]
                probe[event.slot] += 1
                if can_call_using(rules, probe, event.slot, bonus_character=bonus):
                    # Blend toward 1.0 with age: an old pass constrains the hand
                    # that seat *had*, not the one it has now.
                    weight *= 1.0 - (1.0 - PASS_VIOLATION) * recency
            if weight < 1e-12:
                return 1e-12

        # A seat rarely holds what it just threw. `recent_discards` is ordered
        # most-recent-first, so position stands in for age directly.
        for seat, recent in enumerate(state.recent_discards):
            if seat == self.viewer:
                continue
            for age, slot in enumerate(recent):
                if hands[seat][slot] > 0:
                    recency = EVIDENCE_DECAY**age
                    weight *= 1.0 - (1.0 - DISCARD_VIOLATION) * recency

        return max(weight, 1e-12)

    def location(self, particles: list[Particle] | None = None) -> list[list[float]]:
        """Expected count of each slot per seat, plus the deck.

        Returns `players + 1` rows; the last is the draw pile. The viewer's own row
        is their real hand, so this is directly comparable across seats.
        """
        state = self._require_state()
        if particles is None:
            particles = self.sample()

        rows = [[0.0] * self.n_slots for _ in range(len(state.seats) + 1)]
        for p in particles:
            for seat, hand in enumerate(p.hands):
                row = rows[seat]
                for slot in range(self.n_slots):
                    if hand[slot]:
                        row[slot] += p.weight * hand[slot]
            row = rows[-1]
            for slot in range(self.n_slots):
                if p.deck[slot]:
                    row[slot] += p.weight * p.deck[slot]
        return rows

    # -------------------------------------------------------------- misc ----
    def naive_unseen(self) -> list[float]:
        """What the in-game counter implies: every card not seen is still live.

        Kept for comparison rather than use. It ignores that the deck is a *subset*
        of the possible cards, so it overstates availability by roughly the ratio
        of the theoretical pool to the deck. `tests/scenarios/test_belief_scenarios.py`
        measures how much better the posterior does; that gap is the edge.
        """
        return [float(self.cap - self.seen[s]) for s in range(self.n_slots)]

    def _bonus_character(self) -> int | None:
        return self._bonus

    def set_bonus_character(self, bonus: int | None) -> None:
        """The bonus holomem is public, but arrives on GameSetup, not PublicState.

        It only affects payouts, never which hands are legal, so a belief that
        never learns it still reasons correctly about what opponents can claim.
        """
        self._bonus = bonus

    def _require_state(self) -> PublicState:
        if self.state is None:
            raise RuntimeError("belief has not observed a state yet")
        return self.state
