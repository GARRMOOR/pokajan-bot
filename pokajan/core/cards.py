"""Card space: the mapping between cards and count-vector slots.

A card is a (character, colour) pair. Because the deck holds up to 3 identical
copies of each pair, and because nothing in the game ever distinguishes one copy
from another, the whole game state is representable as counts rather than lists
of individual cards.

That choice runs through everything downstream: hand evaluation becomes a few
array scans instead of combinatorial enumeration, and the action space becomes
permutation-invariant (you discard "a blue AzKi", never "the third card in my
hand"). Both matter enough that the layout is fixed here once and depended on
everywhere else.

Slot indices are dense and stable for a given roster:

    slot = character_index * n_colors + color_index

so a roster of 17 characters over 3 colours occupies slots 0..50. The roster is
redrawn every game and varies from 14 to 19 characters, so *nothing* may assume a
fixed slot count -- always read it from the CardSpace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

# A count vector: one entry per slot, each 0..max_per_color. Plain lists rather
# than numpy — at ~50 elements the numpy call overhead dominates the arithmetic,
# and the engine's inner loop is called tens of millions of times in training.
Counts = list[int]


@dataclass(frozen=True)
class CardSpace:
    """Immutable description of the cards in play for one game.

    Built once at deal time from the rules config and then shared, so the derived
    lookup tables below are computed a single time rather than per decision.
    """

    character_ids: tuple[str, ...]
    character_names: tuple[str, ...]
    colors: tuple[str, ...]
    max_per_color: int

    # group_id -> the character indices belonging to it
    group_ids: tuple[str, ...] = ()
    group_names: tuple[str, ...] = ()
    group_members: tuple[tuple[int, ...], ...] = ()

    # --- derived lookups, all precomputed in __post_init__ -------------------
    # slot -> character index / colour index
    slot_char: tuple[int, ...] = field(default=(), repr=False)
    slot_color: tuple[int, ...] = field(default=(), repr=False)
    # character index -> its slots (one per colour); colour index -> its slots
    char_slots: tuple[tuple[int, ...], ...] = field(default=(), repr=False)
    color_slots: tuple[tuple[int, ...], ...] = field(default=(), repr=False)
    # character index -> the groups containing it (usually exactly one)
    char_groups: tuple[tuple[int, ...], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        n_chars = len(self.character_ids)
        n_colors = len(self.colors)

        slot_char = tuple(c for c in range(n_chars) for _ in range(n_colors))
        slot_color = tuple(k for _ in range(n_chars) for k in range(n_colors))
        char_slots = tuple(
            tuple(c * n_colors + k for k in range(n_colors)) for c in range(n_chars)
        )
        color_slots = tuple(
            tuple(c * n_colors + k for c in range(n_chars)) for k in range(n_colors)
        )
        char_groups = tuple(
            tuple(g for g, members in enumerate(self.group_members) if c in members)
            for c in range(n_chars)
        )

        # frozen dataclass: assign derived fields through object.__setattr__
        object.__setattr__(self, "slot_char", slot_char)
        object.__setattr__(self, "slot_color", slot_color)
        object.__setattr__(self, "char_slots", char_slots)
        object.__setattr__(self, "color_slots", color_slots)
        object.__setattr__(self, "char_groups", char_groups)

    # ------------------------------------------------------------- sizes ----
    @property
    def n_chars(self) -> int:
        return len(self.character_ids)

    @property
    def n_colors(self) -> int:
        return len(self.colors)

    @property
    def n_slots(self) -> int:
        return len(self.character_ids) * len(self.colors)

    @property
    def n_groups(self) -> int:
        return len(self.group_members)

    # ------------------------------------------------------- construction ---
    @classmethod
    def from_config(cls, characters, groups, colors, max_per_color) -> "CardSpace":
        """Build from the parsed `characters` / `groups` / `deck` config blocks."""
        char_ids = tuple(c["id"] for c in characters)
        index_of = {cid: i for i, cid in enumerate(char_ids)}

        members: list[tuple[int, ...]] = []
        for g in groups:
            missing = [m for m in g["members"] if m not in index_of]
            if missing:
                raise ValueError(
                    f"group {g['id']!r} lists characters not in the roster: {missing}"
                )
            members.append(tuple(index_of[m] for m in g["members"]))

        return cls(
            character_ids=char_ids,
            character_names=tuple(c.get("name", c["id"]) for c in characters),
            colors=tuple(colors),
            max_per_color=int(max_per_color),
            group_ids=tuple(g["id"] for g in groups),
            group_names=tuple(g.get("name", g["id"]) for g in groups),
            group_members=tuple(members),
        )

    # ------------------------------------------------------------ lookups ---
    def slot(self, character: int | str, color: int | str) -> int:
        """Slot index for a (character, colour), accepting ids or indices."""
        c = self.char_index(character)
        k = self.color_index(color)
        return c * self.n_colors + k

    def char_index(self, character: int | str) -> int:
        if isinstance(character, int):
            return character
        try:
            return self.character_ids.index(character)
        except ValueError:
            raise KeyError(f"unknown character {character!r}") from None

    def color_index(self, color: int | str) -> int:
        if isinstance(color, int):
            return color
        try:
            return self.colors.index(color)
        except ValueError:
            raise KeyError(f"unknown colour {color!r}") from None

    def group_index(self, group: int | str) -> int:
        if isinstance(group, int):
            return group
        try:
            return self.group_ids.index(group)
        except ValueError:
            raise KeyError(f"unknown group {group!r}") from None

    # ------------------------------------------------------------ vectors ---
    def zeros(self) -> Counts:
        return [0] * self.n_slots

    def char_totals(self, counts: Counts) -> list[int]:
        """Copies held of each character, summed across colours."""
        n_colors = self.n_colors
        return [sum(counts[i : i + n_colors]) for i in range(0, len(counts), n_colors)]

    def color_totals(self, counts: Counts) -> list[int]:
        """Cards held of each colour, summed across characters."""
        return [sum(counts[s] for s in slots) for slots in self.color_slots]

    def char_totals_in_color(self, counts: Counts, color: int) -> list[int]:
        """Copies of each character held *in one colour* — the monochrome check."""
        n_colors = self.n_colors
        return [counts[c * n_colors + color] for c in range(self.n_chars)]

    def from_pairs(self, pairs: Iterable[tuple[int | str, int | str]]) -> Counts:
        """Count vector from explicit (character, colour) pairs. For tests and fixtures."""
        counts = self.zeros()
        for character, color in pairs:
            counts[self.slot(character, color)] += 1
        return counts

    def describe_slot(self, slot: int) -> str:
        return f"{self.character_names[self.slot_char[slot]]} ({self.colors[self.slot_color[slot]]})"

    def describe(self, counts: Counts) -> str:
        """Human-readable rendering, for logs, test failures and the GUI."""
        parts = [
            f"{self.describe_slot(s)}x{n}" if n > 1 else self.describe_slot(s)
            for s, n in enumerate(counts)
            if n
        ]
        return ", ".join(parts) if parts else "(empty)"


def total(counts: Sequence[int]) -> int:
    """Number of cards in a count vector."""
    return sum(counts)
