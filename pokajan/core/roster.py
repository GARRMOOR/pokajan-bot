"""Building a `Rules` for the roster a particular game actually dealt.

`rules/pokajan_v1.yaml` pins one roster because a config file has to say *something*,
but the real game redraws its characters and groups every round -- 14 to 19 holomem
across four groups, observed at 4/4/3/5, 4/4/4/3 and 4/4/4/5. So the pinned roster is
a development default, not a description of the game, and anything that wants to
advise on a real round has to build its own.

That this is a small module is the point. Everything downstream -- the engine, the
belief, the observation encoder, the agents -- takes its dimensions from
`rules.cards`, so a new roster is a new `Rules` and nothing else changes. If that
were not already true this would be a refactor rather than fifty lines.

What is worth being fussy about is refusing a bad roster. The screen reader will get
this wrong sometimes: a missed character, a portrait matched to the wrong holomem, a
group boundary read one cell short. Every one of those produces a `Rules` that loads
happily and then quietly misprices the game, because the composition posterior is
inferred against a slot space that does not match reality. Advice from a wrong roster
looks exactly like advice from a right one. So the checks below are deliberately
strict, and the module raises rather than repairing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .rules import Rules

# Observed range, kept as a guard rather than a rule. A roster outside it is far more
# likely to be a misread panel than a genuinely unusual game -- and if the real game
# ever does deal outside it, the error message says exactly what was seen.
MIN_CHARACTERS, MAX_CHARACTERS = 14, 19
# Four groups of at most five, from the fixed 4x5 grid the game displays.
MAX_GROUP_SIZE = 5


class RosterError(ValueError):
    """A roster that cannot be turned into a playable game."""


@dataclass(frozen=True)
class ObservedGroup:
    """One row of the game's group panel."""

    id: str
    name: str
    members: tuple[str, ...]      # character ids, in the order shown


@dataclass(frozen=True)
class ObservedRoster:
    """What the panel and the bonus card say this round is played with."""

    groups: tuple[ObservedGroup, ...]
    bonus_character: str | None = None
    # Display names, for anything a human reads. Ids are what the recogniser
    # produces; names are cosmetic and are filled in from the base rules or derived
    # from the id when not supplied.
    names: Mapping[str, str] = field(default_factory=dict)

    @property
    def characters(self) -> tuple[str, ...]:
        """Every character, in group order. This defines the slot layout."""
        return tuple(member for group in self.groups for member in group.members)


def rules_for_roster(base: Rules, roster: ObservedRoster) -> Rules:
    """A `Rules` playing `base`'s game with `roster`'s characters.

    Everything that is a rule -- payouts, deck size, hand limit, the coin floor --
    comes from `base` untouched. Only the roster changes, which is the whole point:
    a misread roster must never be able to alter what a hand pays.

    `rules_hash` changes with the roster, as it should. A belief calibrated to a
    17-holomem slot space means nothing in a 15-holomem game.
    """
    _validate(base, roster)

    known = dict(zip(base.cards.character_ids, base.cards.character_names))
    raw = dict(base.raw)
    raw["characters"] = [
        {"id": cid, "name": roster.names.get(cid) or known.get(cid) or _derive_name(cid)}
        for cid in roster.characters
    ]
    raw["groups"] = [
        {"id": group.id, "name": group.name, "members": list(group.members)}
        for group in roster.groups
    ]
    raw["bonus_character"] = roster.bonus_character

    return Rules.from_dict(raw, path=base.path)


def roster_of(rules: Rules) -> ObservedRoster:
    """The inverse, so a `Rules` can stand in for an observation in tests."""
    space = rules.cards
    bonus = (
        space.character_ids[rules.bonus_character]
        if rules.bonus_character is not None else None
    )
    return ObservedRoster(
        groups=tuple(
            ObservedGroup(
                id=space.group_ids[g],
                name=space.group_names[g],
                members=tuple(space.character_ids[c] for c in members),
            )
            for g, members in enumerate(space.group_members)
        ),
        bonus_character=bonus,
        names=dict(zip(space.character_ids, space.character_names)),
    )


# ------------------------------------------------------------- validation ----
def _validate(base: Rules, roster: ObservedRoster) -> None:
    expected_groups = base.cards.n_groups
    if len(roster.groups) != expected_groups:
        raise RosterError(
            f"expected {expected_groups} groups, read {len(roster.groups)}"
        )

    for group in roster.groups:
        if not group.members:
            raise RosterError(f"group {group.id!r} has no members")
        if len(group.members) > MAX_GROUP_SIZE:
            raise RosterError(
                f"group {group.id!r} has {len(group.members)} members, "
                f"more than the {MAX_GROUP_SIZE}-cell row the game displays"
            )

    ids = roster.characters
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicates:
        # A character in two groups would make one card complete two different group
        # hands, which the payout table has no answer for.
        raise RosterError(f"characters appear in more than one group: {duplicates}")

    if not MIN_CHARACTERS <= len(ids) <= MAX_CHARACTERS:
        raise RosterError(
            f"read {len(ids)} characters, outside the {MIN_CHARACTERS}-"
            f"{MAX_CHARACTERS} ever observed: {list(ids)}"
        )

    # The constraint nobody thinks of, and the one that fails least gracefully: the
    # deck has to be buildable. Every (character, colour) slot holds at most
    # max_per_color, so a roster too small cannot supply the fixed deck size at all.
    capacity = len(ids) * base.cards.n_colors * base.cards.max_per_color
    if capacity < base.deck_size:
        raise RosterError(
            f"{len(ids)} characters can hold at most {capacity} cards, "
            f"short of the {base.deck_size}-card deck"
        )

    if roster.bonus_character is not None and roster.bonus_character not in ids:
        raise RosterError(
            f"bonus holomem {roster.bonus_character!r} is not in the roster"
        )


def _derive_name(character_id: str) -> str:
    """A readable name for a character the base config has never heard of.

    The recogniser works from ids, but the advisor writes sentences a human reads, so
    something has to fill the gap until the card art in data/cards/ can supply a
    proper catalogue. "gawr_gura" reads as "Gawr Gura", which is right often enough
    to be better than showing the id.
    """
    return " ".join(part.capitalize() for part in character_id.split("_") if part)
