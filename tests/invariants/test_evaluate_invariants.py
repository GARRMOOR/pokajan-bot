"""Properties of hand evaluation that must hold under any rules config.

The theme here is that a Call must be *payable from the hand it came from*. A call
that spends cards you do not hold, or spends the wrong number of them, would show
up in training as free coins — and a policy will find that bug far faster than a
human reading the code will.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pokajan.core.evaluate import best_call, can_call, enumerate_calls
from pokajan.core.rules import HandKind, Rules
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

SETTINGS = settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@st.composite
def rules_and_hand(draw, max_cards: int = 8):
    """A config plus a hand that could plausibly arise under it."""
    rules = Rules.from_dict(draw(rules_configs()))
    space = rules.cards

    counts = space.zeros()
    n_cards = draw(st.integers(min_value=0, max_value=max_cards))
    for _ in range(n_cards):
        slot = draw(st.integers(min_value=0, max_value=space.n_slots - 1))
        if counts[slot] < space.max_per_color:
            counts[slot] += 1
    return rules, counts


@st.composite
def rules_and_scoring_hand(draw):
    """A hand guaranteed to contain at least one scoring shape.

    Random hands rarely score, so the properties that only bite on a real Call
    would almost never be exercised without this.
    """
    rules = Rules.from_dict(draw(rules_configs()))
    space = rules.cards
    counts = space.zeros()

    if draw(st.booleans()):
        # Plant a triple.
        c = draw(st.integers(min_value=0, max_value=space.n_chars - 1))
        slots = space.char_slots[c]
        if draw(st.booleans()):
            counts[slots[draw(st.integers(0, space.n_colors - 1))]] = 3
        else:
            for k in range(3):
                counts[slots[k % space.n_colors]] += 1
    else:
        # Plant a group.
        g = draw(st.integers(min_value=0, max_value=space.n_groups - 1))
        mono = draw(st.booleans())
        shared = draw(st.integers(0, space.n_colors - 1))
        for m in space.group_members[g]:
            k = shared if mono else draw(st.integers(0, space.n_colors - 1))
            counts[space.char_slots[m][k]] += 1

    return rules, counts


@given(rules_and_hand())
@SETTINGS
def test_can_call_agrees_with_best_call(args):
    rules, counts = args
    bonus = rules.bonus_character
    assert can_call(rules, counts, bonus_character=bonus) == (
        best_call(rules, counts, bonus_character=bonus) is not None
    )


@given(rules_and_scoring_hand())
@SETTINGS
def test_a_call_is_payable_from_the_hand(args):
    rules, counts = args
    for call in enumerate_calls(
        rules, counts, bonus_character=rules.bonus_character, all_selections=True
    ):
        assert len(call.cards) == len(counts)
        for slot, spend in enumerate(call.cards):
            assert 0 <= spend <= counts[slot], (
                f"call spends {spend} of slot {slot} but the hand holds {counts[slot]}"
            )


@given(rules_and_scoring_hand())
@SETTINGS
def test_calls_spend_the_right_number_of_cards(args):
    rules, counts = args
    for call in enumerate_calls(
        rules, counts, bonus_character=rules.bonus_character, all_selections=True
    ):
        spent = sum(call.cards)
        if call.kind is HandKind.TRIPLE:
            assert spent == 3
        else:
            assert spent == len(rules.cards.group_members[call.group])


@given(rules_and_scoring_hand())
@SETTINGS
def test_triples_are_one_character_and_groups_are_one_of_each(args):
    rules, counts = args
    space = rules.cards
    for call in enumerate_calls(
        rules, counts, bonus_character=rules.bonus_character, all_selections=True
    ):
        chars_used = {space.slot_char[s] for s, n in enumerate(call.cards) if n}
        if call.kind is HandKind.TRIPLE:
            assert chars_used == {call.character}
        else:
            members = set(space.group_members[call.group])
            assert chars_used == members
            # Exactly one copy of each member: a group hand can never double up.
            totals = space.char_totals(list(call.cards))
            assert all(totals[m] == 1 for m in members)


@given(rules_and_scoring_hand())
@SETTINGS
def test_monochrome_flag_matches_the_cards_spent(args):
    rules, counts = args
    space = rules.cards
    for call in enumerate_calls(
        rules, counts, bonus_character=rules.bonus_character, all_selections=True
    ):
        colors_used = {space.slot_color[s] for s, n in enumerate(call.cards) if n}
        if call.monochrome:
            assert colors_used == {call.color}
        else:
            # A mixed-flagged call must genuinely be mixed, or it would be
            # underpaid — and underpayment is the direction that silently costs
            # the agent coins rather than crashing.
            assert len(colors_used) > 1


@given(rules_and_scoring_hand())
@SETTINGS
def test_bonus_flag_matches_the_characters_spent(args):
    rules, counts = args
    space = rules.cards
    bonus = rules.bonus_character
    for call in enumerate_calls(rules, counts, bonus_character=bonus, all_selections=True):
        chars_used = {space.slot_char[s] for s, n in enumerate(call.cards) if n}
        assert call.bonus == (bonus is not None and bonus in chars_used)


@given(rules_and_scoring_hand())
@SETTINGS
def test_best_call_is_the_highest_paying_call(args):
    rules, counts = args
    calls = enumerate_calls(rules, counts, bonus_character=rules.bonus_character)
    top = best_call(rules, counts, bonus_character=rules.bonus_character)
    assert top is not None
    assert top.payout == max(c.payout for c in calls)
    # And the list itself is ordered, which the GUI's alternatives panel relies on.
    assert [c.payout for c in calls] == sorted((c.payout for c in calls), reverse=True)


@given(rules_and_scoring_hand())
@SETTINGS
def test_payout_matches_a_fresh_recomputation(args):
    """Guards against enumerate_calls drifting from the rules table."""
    rules, counts = args
    for call in enumerate_calls(rules, counts, bonus_character=rules.bonus_character):
        expected = rules.payout(
            call.kind,
            group_size=None if call.group is None else len(rules.cards.group_members[call.group]),
            monochrome=call.monochrome,
            bonus=call.bonus,
            claimed=call.claimed,
        )
        assert call.payout == expected


@given(rules_and_scoring_hand())
@SETTINGS
def test_all_selections_is_a_superset(args):
    rules, counts = args
    bonus = rules.bonus_character
    default = enumerate_calls(rules, counts, bonus_character=bonus)
    everything = enumerate_calls(rules, counts, bonus_character=bonus, all_selections=True)
    assert len(everything) >= len(default)
    # The default view must not hide a payout that the full view can reach, or an
    # agent using the cheap path would leave coins on the table.
    assert max((c.payout for c in everything), default=0) == max(
        (c.payout for c in default), default=0
    )


@given(rules_and_hand())
@SETTINGS
def test_empty_and_small_hands_never_score(args):
    rules, counts = args
    if sum(counts) < min(
        3, min(len(m) for m in rules.cards.group_members)
    ):
        assert best_call(rules, counts, bonus_character=rules.bonus_character) is None
