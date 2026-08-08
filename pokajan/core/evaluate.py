"""Finding the scoring hands in a count vector.

Two facts from the real game make this much cheaper than it looks:

  * only one hand may score at a time, and
  * a card may only be used in one call.

So there is never a set-cover problem to solve. Scoring is just "the single
highest-paying legal call", and because the confirmed tiebreak is by coin payout,
that same number *is* the hand's strength when several players claim the same
discard. One function answers scoring, tiebreak and the GUI hint panel.

The split between `enumerate_calls` and `best_call` is deliberate. Which copies a
mixed-colour hand consumes does not change what it pays, but it does change what
is left behind — spending your only blue AzKi can cost a monochrome hand later.
That is a *policy* judgement, not a rule, so the rules layer enumerates and the
agent layer chooses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .cards import CardSpace, Counts
from .rules import HandKind, Rules


@dataclass(frozen=True)
class Call:
    """One legal scoring hand, with the exact cards it consumes."""

    kind: HandKind
    payout: int
    cards: tuple[int, ...]           # count vector of the cards spent
    monochrome: bool
    bonus: bool
    claimed: bool
    character: int | None = None     # triples
    group: int | None = None         # group hands
    color: int | None = None         # the colour, when monochrome

    # Sort key for deterministic tiebreaking between equal-paying calls. Never
    # used to decide *strength* — that is `payout` alone.
    _order: tuple = field(default=(), repr=False, compare=False)

    def describe(self, space: CardSpace) -> str:
        if self.kind is HandKind.TRIPLE:
            what = f"triple {space.character_names[self.character]}"
        else:
            what = f"group {space.group_names[self.group]}"
        tags = []
        if self.monochrome:
            tags.append(f"mono {space.colors[self.color]}")
        if self.bonus:
            tags.append("bonus")
        if self.claimed:
            tags.append("claimed")
        suffix = f" [{', '.join(tags)}]" if tags else ""
        return f"{what}{suffix} = {self.payout}"


def _selections(counts: Counts, slots: tuple[int, ...], need: int) -> list[tuple[int, ...]]:
    """All distinct ways to take `need` cards from one character's colour slots.

    Small by construction: at most 3 colours capped at 3 copies each, so this
    returns a handful of options, never an explosion.
    """
    out: list[tuple[int, ...]] = []
    caps = [counts[s] for s in slots]

    def walk(i: int, left: int, taken: list[int]) -> None:
        if left == 0:
            out.append(tuple(taken))
            return
        if i == len(slots):
            return
        # Take as many as possible from this colour first, so the canonical (first)
        # selection drains the most plentiful colour and preserves singletons —
        # singletons are what group hands need.
        for take in range(min(caps[i], left), -1, -1):
            walk(i + 1, left - take, taken + [take])

    walk(0, need, [])
    return out


def enumerate_calls(
    rules: Rules,
    counts: Counts,
    *,
    bonus_character: int | None,
    claimed: bool = False,
    all_selections: bool = False,
) -> list[Call]:
    """Every legal call available from `counts`, highest-paying first.

    By default one representative per payout-distinct hand is returned, using the
    canonical card selection. Pass `all_selections=True` to also get the
    alternative ways of paying for the same hand, which is what an agent wants
    when deciding which copies to spend.
    """
    space = rules.cards
    n_colors = space.n_colors
    calls: list[Call] = []

    def add(kind, payout, cards, mono, bonus, character=None, group=None, color=None):
        calls.append(
            Call(
                kind=kind,
                payout=payout,
                cards=tuple(cards),
                monochrome=mono,
                bonus=bonus,
                claimed=claimed,
                character=character,
                group=group,
                color=color,
                _order=(-payout, 0 if kind is HandKind.TRIPLE else 1,
                        character if character is not None else -1,
                        group if group is not None else -1,
                        color if color is not None else -1),
            )
        )

    # ------------------------------------------------------------ triples ---
    char_totals = space.char_totals(counts)
    for c, held in enumerate(char_totals):
        if held < 3:
            continue
        slots = space.char_slots[c]
        is_bonus = bonus_character is not None and c == bonus_character

        # Monochrome triple: three copies in a single colour.
        mono_colors = [k for k in range(n_colors) if counts[slots[k]] >= 3]
        for k in mono_colors:
            cards = [0] * len(counts)
            cards[slots[k]] = 3
            add(
                HandKind.TRIPLE,
                rules.payout(HandKind.TRIPLE, monochrome=True, bonus=is_bonus, claimed=claimed),
                cards, True, is_bonus, character=c, color=k,
            )

        # Mixed triple. Only worth listing if it is not simply a worse way of
        # spending the same cards as an available monochrome one.
        mixed = _selections(counts, slots, 3)
        mixed = [sel for sel in mixed if max(sel) < 3]
        if mixed:
            payout = rules.payout(HandKind.TRIPLE, monochrome=False, bonus=is_bonus, claimed=claimed)
            for sel in (mixed if all_selections else mixed[:1]):
                cards = [0] * len(counts)
                for k, take in enumerate(sel):
                    cards[slots[k]] = take
                add(HandKind.TRIPLE, payout, cards, False, is_bonus, character=c)

    # ------------------------------------------------------------- groups ---
    for g, members in enumerate(space.group_members):
        if not all(char_totals[m] >= 1 for m in members):
            continue
        size = len(members)
        is_bonus = bonus_character is not None and bonus_character in members

        # Monochrome group: every member available in one shared colour.
        mono_colors = [
            k for k in range(n_colors)
            if all(counts[space.char_slots[m][k]] >= 1 for m in members)
        ]
        for k in mono_colors:
            cards = [0] * len(counts)
            for m in members:
                cards[space.char_slots[m][k]] = 1
            add(
                HandKind.GROUP,
                rules.payout(HandKind.GROUP, group_size=size, monochrome=True,
                             bonus=is_bonus, claimed=claimed),
                cards, True, is_bonus, group=g, color=k,
            )

        # Mixed group: one copy of each member, colours free. The full product is
        # only enumerated on request — it is up to 3^5 for a five-member group and
        # every entry pays the same.
        payout = rules.payout(HandKind.GROUP, group_size=size, monochrome=False,
                              bonus=is_bonus, claimed=claimed)
        per_member = [
            [k for k in range(n_colors) if counts[space.char_slots[m][k]] >= 1]
            for m in members
        ]
        if all_selections:
            choices = product(*per_member)
        else:
            # Canonical pick: the most plentiful colour for each member, which
            # spends duplicates before singletons.
            choices = [tuple(max(ks, key=lambda k: counts[space.char_slots[m][k]])
                             for m, ks in zip(members, per_member))]
        for combo in choices:
            if len(set(combo)) == 1 and combo[0] in mono_colors:
                continue  # already listed above, and it pays more there
            cards = [0] * len(counts)
            for m, k in zip(members, combo):
                cards[space.char_slots[m][k]] += 1
            add(HandKind.GROUP, payout, cards, False, is_bonus, group=g)

    calls.sort(key=lambda call: call._order)
    return calls


def best_call(
    rules: Rules,
    counts: Counts,
    *,
    bonus_character: int | None,
    claimed: bool = False,
) -> Call | None:
    """The highest-paying legal call, or None if the hand cannot score.

    Ties break deterministically (triples before groups, then by index) so that
    replays and seeded games reproduce exactly. Real ties between *players* are
    resolved by turn order in the engine, not here.
    """
    calls = enumerate_calls(rules, counts, bonus_character=bonus_character, claimed=claimed)
    return calls[0] if calls else None


def can_call(rules: Rules, counts: Counts, *, bonus_character: int | None) -> bool:
    """Cheap yes/no test, for masking the CALL action.

    Short-circuits rather than building Call objects, because the engine asks this
    of every seat on every discard.
    """
    space = rules.cards
    char_totals = space.char_totals(counts)
    if any(t >= 3 for t in char_totals):
        return True
    return any(
        all(char_totals[m] >= 1 for m in members)
        for members in space.group_members
    )
