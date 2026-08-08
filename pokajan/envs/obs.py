"""Turning a seat's view of the game into a vector.

Three commitments here, all of them expensive to change later:

**Dimensions come from the config, never from a constant.** The roster is redrawn
every game and runs from 14 to 19 characters, so the layout is computed from the
`CardSpace` and exposed as `ObsSpec`. A checkpoint is only valid for the spec it
was trained under, which is why `ObsSpec.signature` exists and gets stamped
alongside the rules hash.

**Everything is viewer-relative.** Opponents appear in turn order starting from
the seat after the viewer, never by absolute seat index. Nothing in Pokajan
depends on which chair you are sitting in, so encoding absolute seats would make
the network learn the same strategy four times over.

**The legality mask is not in here.** It is a hard constraint applied to the
policy's logits, not a feature to be inferred. Mixing the two invites a network
that has learned illegal moves are merely unpromising.

Values are plain floats in a list rather than an array. numpy is a training-only
dependency (see requirements-train.txt) and this module is imported by the
heuristic agent and the GUI, which run on the laptop with neither numpy nor torch
installed. Batching into arrays is the training loop's job at M5.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.actions import DecisionType
from ..core.rules import Rules
from ..core.state import Phase
from ..server.protocol import PublicState
from .belief import Belief

# How fast a seat's discard history fades in the `opp_recent` block. Ordering
# carries the read on what a player is collecting, and the newest throw carries
# most of it.
RECENCY_DECAY = 0.7

# Divisor for call counts, so the feature lands in roughly [0, 1] without clipping
# a long chain to a constant.
CALLS_SCALE = 5.0


@dataclass(frozen=True)
class Block:
    name: str
    offset: int
    size: int


@dataclass(frozen=True)
class ObsSpec:
    """Fixed layout for one rules config."""

    rules: Rules
    blocks: tuple[Block, ...]
    size: int
    signature: str

    @classmethod
    def build(cls, rules: Rules) -> "ObsSpec":
        space = rules.cards
        n = space.n_slots
        opponents = rules.play.players - 1

        shape = [
            # --- what this seat holds -------------------------------------
            ("hand", n),
            ("hand_char_totals", space.n_chars),
            ("hand_color_totals", space.n_colors),
            ("group_progress", space.n_groups * 2),
            # --- what is on the table -------------------------------------
            ("claimable", n),
            ("table", n),
            ("scored", n),
            # --- belief ---------------------------------------------------
            ("composition_mean", n),
            ("composition_std", n),
            ("unseen_mean", n),
            ("deck_mean", n),
            ("opponent_hands", opponents * n),
            # --- what opponents have done ---------------------------------
            ("opponent_discards", opponents * n),
            ("opponent_recent", opponents * n),
            ("opponent_scalars", opponents * 6),
            # --- static for the game --------------------------------------
            ("group_membership", space.n_chars * space.n_groups),
            ("bonus_character", space.n_chars),
            # --- where we are ---------------------------------------------
            ("own_scalars", 5),
            ("phase", len(Phase)),
            ("decision", len(DecisionType)),
            ("globals", 4),
        ]

        blocks, offset = [], 0
        for name, size in shape:
            blocks.append(Block(name=name, offset=offset, size=size))
            offset += size

        signature = (
            f"obs_v1:{space.n_chars}c{space.n_colors}k{space.n_groups}g"
            f"{rules.play.players}p:{offset}"
        )
        return cls(rules=rules, blocks=tuple(blocks), size=offset, signature=signature)

    def block(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(f"no observation block named {name!r}")

    def view(self, vector: list[float], name: str) -> list[float]:
        """The slice of an encoded vector belonging to one block. For tests and the GUI."""
        b = self.block(name)
        return vector[b.offset : b.offset + b.size]

    def layout(self) -> list[tuple[str, int, int]]:
        return [(b.name, b.offset, b.size) for b in self.blocks]


def encode(
    spec: ObsSpec,
    state: PublicState,
    *,
    belief: Belief | None = None,
    particles: list | None = None,
    decision: DecisionType | None = None,
    claimable_slot: int | None = None,
    bonus_character: int | None = None,
    risk_alpha: float = 0.0,
) -> list[float]:
    """One seat's view as a vector of length `spec.size`.

    `belief` is optional so the encoder can be exercised without paying for
    inference; the belief blocks are left at zero when it is absent. Training
    always supplies one — those blocks are the point. Drawing particles is the
    expensive part, so `particles` lets a caller that has already sampled reuse
    them rather than paying for a second draw.

    `bonus_character` is passed in rather than read off the state because it is
    drawn per game and travels on `GameSetup`, not `PublicState`. It is public
    information; it simply is not repeated on every message.
    """
    rules = spec.rules
    space = rules.cards
    n = space.n_slots
    cap = float(space.max_per_color)
    players = rules.play.players
    viewer = state.viewer
    hand_limit = float(rules.play.hand_limit)
    coins_scale = float(rules.play.initial_coins)

    out = [0.0] * spec.size

    def put(name: str, values, *, at: int = 0) -> None:
        b = spec.block(name)
        base = b.offset + at
        for i, v in enumerate(values):
            out[base + i] = v

    # -------------------------------------------------------- own holdings --
    put("hand", [state.hand[s] / cap for s in range(n)])
    # Divided by 3 rather than the hand limit: three copies *is* a triple, so the
    # feature saturates exactly where the hand stops improving.
    put("hand_char_totals", [t / 3.0 for t in space.char_totals(state.hand)])
    put("hand_color_totals", [t / hand_limit for t in space.color_totals(state.hand)])

    # For each group: how much of it this seat holds, mixed and best-monochrome.
    # Derivable from `hand`, but only via a comparison across slots that a plain
    # MLP would have to spend capacity learning, and it is the single most
    # decision-relevant summary of a hand.
    progress = []
    char_totals = space.char_totals(state.hand)
    for members in space.group_members:
        size = len(members)
        progress.append(sum(1 for m in members if char_totals[m] > 0) / size)
        best = 0
        for k in range(space.n_colors):
            held = sum(1 for m in members if state.hand[space.char_slots[m][k]] > 0)
            best = max(best, held)
        progress.append(best / size)
    put("group_progress", progress)

    # ---------------------------------------------------------- the table ---
    if claimable_slot is not None:
        put("claimable", [1.0], at=claimable_slot)
    put("table", [state.table[s] / cap for s in range(n)])
    put("scored", [state.scored[s] / cap for s in range(n)])

    # -------------------------------------------------------------- belief --
    if belief is not None:
        mean = belief.expected_composition()
        var = belief.composition_variance()
        unseen = belief.expected_unseen()
        put("composition_mean", [m / cap for m in mean])
        put("composition_std", [v**0.5 / cap for v in var])
        put("unseen_mean", [u / cap for u in unseen])

        # Sampling dominates the cost of encoding, so callers that already drew
        # particles for their own reasons — the heuristic's discard-danger scan,
        # PIMC's determinizations — pass them straight through instead of paying
        # twice.
        location = belief.location(particles if particles is not None else belief.sample())
        put("deck_mean", [location[players][s] / cap for s in range(n)])
        for i, seat in enumerate(_opponent_order(viewer, players)):
            put("opponent_hands", [location[seat][s] / cap for s in range(n)], at=i * n)

    # ------------------------------------------------------ opponent play ---
    for i, seat in enumerate(_opponent_order(viewer, players)):
        view = state.seats[seat]
        put("opponent_discards", [d / cap for d in view.discards], at=i * n)

        recent = [0.0] * n
        for age, slot in enumerate(state.recent_discards[seat]):
            recent[slot] += RECENCY_DECAY**age
        put("opponent_recent", recent, at=i * n)

        put(
            "opponent_scalars",
            [
                view.coins / coins_scale,
                view.hand_size / hand_limit,
                view.calls_made / CALLS_SCALE,
                view.coins_won / coins_scale,
                view.coins_paid / coins_scale,
                1.0 if seat == state.current_seat else 0.0,
            ],
            at=i * 6,
        )

    # -------------------------------------------------------------- static --
    membership = []
    for c in range(space.n_chars):
        groups = space.char_groups[c]
        membership.extend(1.0 if g in groups else 0.0 for g in range(space.n_groups))
    put("group_membership", membership)

    bonus = bonus_character if bonus_character is not None else rules.bonus_character
    if bonus is not None:
        put("bonus_character", [1.0], at=bonus)

    # ------------------------------------------------------------- context --
    me = state.seats[viewer]
    put(
        "own_scalars",
        [
            me.coins / coins_scale,
            me.hand_size / hand_limit,
            me.calls_made / CALLS_SCALE,
            me.coins_won / coins_scale,
            me.coins_paid / coins_scale,
        ],
    )

    put("phase", [1.0 if p.name == state.phase else 0.0 for p in Phase])
    if decision is not None:
        put("decision", [1.0 if d is decision else 0.0 for d in DecisionType])

    deck_frac = state.deck_remaining / float(rules.deck_size)
    put(
        "globals",
        [
            deck_frac,
            # Turns until my turn comes round again, which is what actually gates
            # calling a hand that completed off-turn.
            ((state.current_seat - viewer) % players) / float(players),
            risk_alpha,
            1.0,
        ],
    )

    return out


def _opponent_order(viewer: int, players: int) -> list[int]:
    """Opponents in turn order starting from the seat after the viewer."""
    return [(viewer + i) % players for i in range(1, players)]
