"""Building a game around the roster a real round actually dealt.

Two things are being checked, and they pull in opposite directions.

The first is that a roster really is the only thing that changes: a 14-holomem game
and a 19-holomem one must both deal, play and pay out with no other edit. That claim
is load-bearing for M8 -- if it were false, reading the screen would mean a refactor
rather than a constructor call -- so it is tested by playing games rather than by
inspecting fields.

The second is that a *wrong* roster is refused. The screen reader will misread panels,
and a misread roster produces a `Rules` that loads happily and then misprices the
whole game, because the belief is inferred against a slot space that does not match
reality. There is no symptom to notice: advice from a wrong roster looks exactly like
advice from a right one.
"""

from __future__ import annotations

import pytest

from pokajan.agents.base import GreedyCallerAgent
from pokajan.core.engine import Engine
from pokajan.core.roster import (
    MAX_GROUP_SIZE,
    ObservedGroup,
    ObservedRoster,
    RosterError,
    roster_of,
    rules_for_roster,
)
from pokajan.envs.driver import play_game

pytestmark = pytest.mark.invariant


def make_roster(sizes, *, bonus=None, groups=4):
    """A synthetic roster with `sizes` members per group."""
    made, n = [], 0
    for g in range(groups):
        members = tuple(f"holomem_{n + i:02d}" for i in range(sizes[g]))
        n += sizes[g]
        made.append(ObservedGroup(id=f"g{g}", name=f"Group {g}", members=members))
    return ObservedRoster(groups=tuple(made), bonus_character=bonus)


# ------------------------------------------------------------ construction ---

def test_a_rules_round_trips_through_its_own_roster(real_rules):
    """`roster_of` and `rules_for_roster` are inverses, hash included.

    Which makes an observed roster substitutable for a config one everywhere, and
    means the rest of this file can build games the same way M8 will.
    """
    rebuilt = rules_for_roster(real_rules, roster_of(real_rules))

    assert rebuilt.rules_hash == real_rules.rules_hash
    assert rebuilt.cards.character_ids == real_rules.cards.character_ids
    assert rebuilt.cards.group_members == real_rules.cards.group_members


@pytest.mark.parametrize("sizes", [(4, 4, 3, 3), (4, 4, 3, 5), (4, 4, 4, 5), (5, 5, 5, 4)])
def test_every_observed_roster_shape_deals_plays_and_pays(real_rules, sizes):
    """The claim that nothing downstream cares about roster size, tested by playing.

    Group sizes here are the ones seen in real games plus the extremes of the range.
    """
    rules = rules_for_roster(real_rules, make_roster(sizes))
    assert rules.cards.n_chars == sum(sizes)

    engine = Engine.new_game(rules, seed=4)
    result = play_game(
        engine, [GreedyCallerAgent(rules, seed=s) for s in range(rules.play.players)]
    )

    assert engine.finished
    assert len(result.final_coins) == rules.play.players
    assert sum(result.final_coins) == (
        rules.play.players * rules.play.initial_coins + engine.state.coins_minted
    )


def test_only_the_roster_changes(real_rules):
    """A misread roster must not be able to alter what a hand pays.

    Everything that is a rule is carried over from the base config by reference to
    the same parsed values, so this is really asserting that the constructor edits
    three keys and nothing else.
    """
    rules = rules_for_roster(real_rules, make_roster((4, 4, 4, 5)))

    assert rules.deck_size == real_rules.deck_size
    assert rules._table_triple == real_rules._table_triple
    assert rules._table_group == real_rules._table_group
    assert rules.bonus_per_copy == real_rules.bonus_per_copy
    assert rules.play == real_rules.play
    assert rules.end == real_rules.end


def test_the_rules_hash_changes_with_the_roster(real_rules):
    """A belief calibrated to 17 holomem means nothing in a 15-holomem game.

    The hash is what stops a checkpoint being loaded against the wrong one.
    """
    small = rules_for_roster(real_rules, make_roster((4, 4, 3, 3)))
    large = rules_for_roster(real_rules, make_roster((4, 4, 4, 5)))

    assert len({small.rules_hash, large.rules_hash, real_rules.rules_hash}) == 3


def test_names_come_from_the_config_when_it_knows_them(real_rules):
    """And are derived from the id when it does not, since the advisor writes prose."""
    known = real_rules.cards.character_ids[0]
    known_name = real_rules.cards.character_names[0]

    roster = ObservedRoster(
        groups=(
            ObservedGroup("a", "A", (known, "gawr_gura", "named_one", "a3")),
            ObservedGroup("b", "B", tuple(f"b{i}" for i in range(4))),
            ObservedGroup("c", "C", tuple(f"c{i}" for i in range(4))),
            ObservedGroup("d", "D", tuple(f"d{i}" for i in range(4))),
        ),
        names={"named_one": "Explicitly Named"},
    )
    space = rules_for_roster(real_rules, roster).cards
    names = dict(zip(space.character_ids, space.character_names))

    assert names[known] == known_name               # from the base config
    assert names["gawr_gura"] == "Gawr Gura"        # derived from the id
    assert names["named_one"] == "Explicitly Named" # supplied by the caller


# -------------------------------------------------------------- refusals -----

def test_the_wrong_number_of_groups_is_refused(real_rules):
    with pytest.raises(RosterError, match="groups"):
        rules_for_roster(real_rules, make_roster((5, 5, 5), groups=3))


def test_a_group_wider_than_its_row_is_refused(real_rules):
    """The panel is a fixed 4x5 grid, so six members means the row was misread."""
    with pytest.raises(RosterError, match=str(MAX_GROUP_SIZE)):
        rules_for_roster(real_rules, make_roster((6, 4, 4, 4)))


def test_a_character_in_two_groups_is_refused(real_rules):
    """One card would complete two different group hands, which nothing prices."""
    roster = make_roster((4, 4, 4, 4))
    clash = ObservedGroup("g0", "Group 0", roster.groups[1].members)
    broken = ObservedRoster(groups=(clash,) + roster.groups[1:])

    with pytest.raises(RosterError, match="more than one group"):
        rules_for_roster(real_rules, broken)


@pytest.mark.parametrize("sizes", [(3, 3, 3, 4), (5, 5, 5, 5)])
def test_a_roster_outside_the_observed_range_is_refused(real_rules, sizes):
    """13 or 20 holomem is far likelier to be a misread panel than a strange game."""
    with pytest.raises(RosterError, match="ever observed"):
        rules_for_roster(real_rules, make_roster(sizes))


def test_a_bonus_holomem_who_is_not_playing_is_refused(real_rules):
    with pytest.raises(RosterError, match="not in the roster"):
        rules_for_roster(real_rules, make_roster((4, 4, 4, 5), bonus="somebody_else"))


def test_a_roster_too_small_to_hold_the_deck_is_refused(real_rules):
    """The constraint nobody thinks of, and the one that fails least gracefully.

    Each (character, colour) slot holds at most three cards, so a 100-card deck needs
    at least twelve holomem to exist at all. Below that the deck builder has nothing
    sensible to do, and what it does instead is not worth finding out from a bug
    report. Checked with the range guard relaxed, since 12 is already below it — the
    two limits are independent and this one is arithmetic rather than observation.
    """
    import pokajan.core.roster as roster_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(roster_module, "MIN_CHARACTERS", 1)
        with pytest.raises(RosterError, match="short of the"):
            rules_for_roster(real_rules, make_roster((3, 3, 3, 2)))
