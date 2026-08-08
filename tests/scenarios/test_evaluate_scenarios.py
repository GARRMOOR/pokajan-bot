"""Golden hands against the frozen fixture rules.

Fixture recap (tests/fixtures/rules_v1.yaml), so the expected numbers below can be
checked by hand:

    characters   a b c d e f          colours  blue orange pink
    groups       left  = {a, b}       right = {c, d, e, f}
    bonus        c
    triple       100
    group        size 2 -> 50,  size 3 -> 200,  size 4 -> 400
    monochrome   x2      bonus x3     (multiplicative)

Note the deliberately awkward choice that a two-member group pays *less* than a
triple. Nothing in the engine may assume groups outrank triples; the tiebreak is
by payout and nothing else, and these tests are what hold that line.
"""

from __future__ import annotations

import pytest

from pokajan.core.evaluate import best_call, can_call, enumerate_calls
from pokajan.core.rules import HandKind

pytestmark = pytest.mark.scenario


def hand(rules, *pairs):
    return rules.cards.from_pairs(pairs)


def top(rules, counts, claimed=False):
    return best_call(
        rules, counts, bonus_character=rules.bonus_character, claimed=claimed
    )


# ------------------------------------------------------------------ triples ---

def test_mixed_triple_pays_base(fixture_rules):
    r = fixture_rules
    call = top(r, hand(r, ("a", "blue"), ("a", "orange"), ("a", "pink")))
    assert call.kind is HandKind.TRIPLE
    assert call.character == r.cards.char_index("a")
    assert call.monochrome is False
    assert call.bonus is False
    assert call.payout == 100


def test_monochrome_triple_doubles(fixture_rules):
    r = fixture_rules
    call = top(r, hand(r, ("a", "blue"), ("a", "blue"), ("a", "blue")))
    assert call.monochrome is True
    assert call.color == r.cards.color_index("blue")
    assert call.payout == 200


def test_bonus_character_triple_triples(fixture_rules):
    """c is the bonus character: 100 x 3."""
    r = fixture_rules
    call = top(r, hand(r, ("c", "blue"), ("c", "orange"), ("c", "pink")))
    assert call.bonus is True
    assert call.payout == 300


def test_monochrome_bonus_triple_stacks(fixture_rules):
    """Both modifiers apply: 100 x 2 x 3."""
    r = fixture_rules
    call = top(r, hand(r, ("c", "pink"), ("c", "pink"), ("c", "pink")))
    assert call.monochrome and call.bonus
    assert call.payout == 600


def test_four_of_a_kind_scores_only_the_triple_inside_it(fixture_rules):
    """Confirmed rule: a fourth copy is worth nothing."""
    r = fixture_rules
    four = hand(r, ("a", "blue"), ("a", "blue"), ("a", "blue"), ("a", "orange"))
    call = top(r, four)
    assert call.payout == 200          # the monochrome triple, not more
    assert sum(call.cards) == 3        # and it spends three cards, leaving the spare


# ------------------------------------------------------------------- groups ---

def test_small_group_can_pay_less_than_a_triple(fixture_rules):
    """The case that stops anyone hardcoding "group beats triple"."""
    r = fixture_rules
    counts = hand(
        r,
        ("a", "blue"), ("b", "orange"),                       # the "left" group, 50
        ("d", "blue"), ("d", "orange"), ("d", "pink"),        # a triple, 100
    )
    call = top(r, counts)
    assert call.kind is HandKind.TRIPLE
    assert call.payout == 100

    payouts = {c.kind: c.payout for c in enumerate_calls(r, counts, bonus_character=r.bonus_character)}
    assert payouts[HandKind.GROUP] == 50


def test_four_member_group_beats_a_triple(fixture_rules):
    r = fixture_rules
    counts = hand(r, ("c", "blue"), ("d", "orange"), ("e", "pink"), ("f", "blue"))
    call = top(r, counts)
    assert call.kind is HandKind.GROUP
    assert call.group == r.cards.group_index("right")
    # Contains the bonus character c, so 400 x 3.
    assert call.bonus is True
    assert call.payout == 1200


def test_monochrome_group_stacks_with_bonus(fixture_rules):
    """400 x 2 (mono) x 3 (bonus) — the fixture's biggest hand."""
    r = fixture_rules
    counts = hand(r, ("c", "pink"), ("d", "pink"), ("e", "pink"), ("f", "pink"))
    call = top(r, counts)
    assert call.monochrome and call.bonus
    assert call.color == r.cards.color_index("pink")
    assert call.payout == 2400


def test_group_without_the_bonus_character_gets_no_bonus(fixture_rules):
    r = fixture_rules
    call = top(r, hand(r, ("a", "blue"), ("b", "blue")))
    assert call.group == r.cards.group_index("left")
    assert call.bonus is False
    assert call.monochrome is True
    assert call.payout == 100          # 50 x 2


def test_incomplete_group_does_not_score(fixture_rules):
    r = fixture_rules
    counts = hand(r, ("c", "blue"), ("d", "blue"), ("e", "blue"))   # missing f
    assert top(r, counts) is None
    assert can_call(r, counts, bonus_character=r.bonus_character) is False


def test_duplicates_do_not_substitute_for_a_missing_member(fixture_rules):
    """Two copies of d never stand in for f."""
    r = fixture_rules
    counts = hand(r, ("c", "blue"), ("d", "blue"), ("d", "pink"), ("e", "blue"))
    assert top(r, counts) is None


# -------------------------------------------------------------- selections ---

def test_monochrome_is_preferred_when_both_are_available(fixture_rules):
    """Holding blue-blue-blue-orange, the call should take the three blues."""
    r = fixture_rules
    counts = hand(r, ("e", "blue"), ("e", "blue"), ("e", "blue"), ("e", "orange"))
    call = top(r, counts)
    assert call.monochrome
    assert call.cards[r.cards.slot("e", "blue")] == 3
    assert call.cards[r.cards.slot("e", "orange")] == 0


def test_alternative_selections_are_offered_for_mixed_hands(fixture_rules):
    """Which copies to spend is a policy choice, so the options must be visible."""
    r = fixture_rules
    counts = hand(r, ("a", "blue"), ("a", "blue"), ("a", "orange"), ("a", "pink"))
    everything = enumerate_calls(
        r, counts, bonus_character=r.bonus_character, all_selections=True
    )
    mixed = [c for c in everything if not c.monochrome]
    distinct = {c.cards for c in mixed}
    assert len(distinct) > 1, "a 2/1/1 split has more than one way to pay for a triple"
    assert all(c.payout == 100 for c in mixed), "the choice must not change what it pays"


def test_canonical_selection_preserves_singletons(fixture_rules):
    """Spend duplicates first, so the cards a group hand needs survive.

    Holding a-blue x2 and a-orange, the mixed triple should take both blues rather
    than stripping the lone orange.
    """
    r = fixture_rules
    counts = hand(r, ("a", "blue"), ("a", "blue"), ("a", "orange"))
    call = top(r, counts)
    assert call.cards[r.cards.slot("a", "blue")] == 2
    assert call.cards[r.cards.slot("a", "orange")] == 1


# ------------------------------------------------------------------ claims ---

def test_claiming_is_flagged_on_the_call(fixture_rules):
    """The flag drives who pays, so it has to survive onto the Call."""
    r = fixture_rules
    counts = hand(r, ("a", "blue"), ("a", "orange"), ("a", "pink"))
    assert top(r, counts, claimed=True).claimed is True
    assert top(r, counts, claimed=False).claimed is False


def test_claimed_and_self_drawn_pay_the_same_under_current_config(fixture_rules):
    """Pinned so the M2 payout work has to update this deliberately.

    The fixture sets the `claimed` modifier to 1.0 because we do not yet know
    whether the real game pays differently for a claimed hand. If it does, this
    test is the one that should fail.
    """
    r = fixture_rules
    counts = hand(r, ("a", "blue"), ("a", "orange"), ("a", "pink"))
    assert top(r, counts, claimed=True).payout == top(r, counts, claimed=False).payout
