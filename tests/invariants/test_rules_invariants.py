"""Properties the rules loader must satisfy under any valid config."""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pokajan.core.rules import HandKind, Payer, Rules, canonical_hash
from tests.conftest import build_config, rules_configs

pytestmark = pytest.mark.invariant

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@given(rules_configs())
@SETTINGS
def test_generated_configs_load(cfg):
    rules = Rules.from_dict(cfg)
    space = rules.cards
    assert space.n_slots == space.n_chars * space.n_colors
    assert space.n_groups == 4
    # Every character sits in exactly one group — the loader rejects orphans, and
    # overlapping groups would break the "one card, one call" accounting.
    assert sum(len(m) for m in space.group_members) == space.n_chars


@given(rules_configs())
@SETTINGS
def test_deck_is_buildable(cfg):
    rules = Rules.from_dict(cfg)
    capacity = rules.cards.n_slots * rules.cards.max_per_color
    assert rules.deck_size <= capacity
    assert rules.play.deal_size * rules.play.players <= rules.deck_size


@given(rules_configs())
@SETTINGS
def test_every_group_is_completable(cfg):
    rules = Rules.from_dict(cfg)
    for members in rules.cards.group_members:
        assert 1 <= len(members) <= rules.play.hand_limit


@given(rules_configs())
@SETTINGS
def test_monochrome_never_pays_less_than_mixed(cfg):
    """The monochrome bonus is a bonus.

    Stated as an inequality rather than an equality so it survives the real
    modifier turning out to be a flat add rather than a multiplier.
    """
    rules = Rules.from_dict(cfg)
    plain = rules.payout(HandKind.TRIPLE)
    mono = rules.payout(HandKind.TRIPLE, monochrome=True)
    assert mono >= plain

    for members in rules.cards.group_members:
        size = len(members)
        assert rules.payout(HandKind.GROUP, group_size=size, monochrome=True) >= rules.payout(
            HandKind.GROUP, group_size=size
        )


@given(rules_configs())
@SETTINGS
def test_payouts_are_positive_integers(cfg):
    rules = Rules.from_dict(cfg)
    for mono in (False, True):
        for copies in (0, 1, 3):
            amount = rules.payout(HandKind.TRIPLE, monochrome=mono, bonus_copies=copies)
            assert isinstance(amount, int) and amount > 0


@given(rules_configs())
@SETTINGS
def test_the_bonus_is_additive_per_copy(cfg):
    """Confirmed: +N per copy of the bonus holomem, not a multiplier.

    So a triple of the bonus character earns it three times over. Asserted as
    linearity in the copy count, which a multiplicative model could not satisfy.
    """
    rules = Rules.from_dict(cfg)
    base = rules.payout(HandKind.TRIPLE)
    for copies in (0, 1, 2, 3):
        assert rules.payout(HandKind.TRIPLE, bonus_copies=copies) == (
            base + rules.bonus_per_copy * copies
        )


@given(rules_configs())
@SETTINGS
def test_nothing_assumes_a_fixed_monochrome_ratio(cfg):
    """The premium differs per hand shape, so only the ordering may be relied on.

    In the real table it ranges from 2.67x on a three-member group to 7x on a
    triple. Any code inferring one ratio from another would be wrong.
    """
    rules = Rules.from_dict(cfg)
    triple_ratio = rules.payout(HandKind.TRIPLE, monochrome=True) / rules.payout(
        HandKind.TRIPLE
    )
    assert triple_ratio >= 1.0
    for members in rules.cards.group_members:
        size = len(members)
        ratio = rules.payout(HandKind.GROUP, group_size=size, monochrome=True) / rules.payout(
            HandKind.GROUP, group_size=size
        )
        assert ratio >= 1.0


@given(rules_configs())
@SETTINGS
def test_payer_follows_the_claim(cfg):
    """Confirmed rule: claiming bills the discarder alone, otherwise the rest split."""
    rules = Rules.from_dict(cfg)
    assert rules.payer_for(claimed=True) is Payer.DISCARDER
    assert rules.payer_for(claimed=False) is Payer.SPLIT_OTHERS


@given(
    n_chars=st.integers(min_value=14, max_value=19),
    reorder=st.booleans(),
)
@SETTINGS
def test_rules_hash_tracks_values_not_layout(n_chars, reorder):
    """Key order must not change the hash; a payout change must.

    Checkpoints are gated on this hash. If it moved every time the file was
    reformatted we would retrain for nothing; if it failed to move on a payout
    edit we would silently evaluate a policy against rules it never saw.
    """
    sizes = [n_chars - 3, 1, 1, 1]
    cfg = build_config(n_chars, sizes)
    baseline = canonical_hash(cfg)

    if reorder:
        shuffled = dict(reversed(list(cfg.items())))
        assert canonical_hash(shuffled) == baseline

    changed = build_config(n_chars, sizes, triple_payout=999)
    assert canonical_hash(changed) != baseline


def test_every_real_payout_splits_evenly_between_the_other_players(real_rules):
    """Confirmed: no real payout is indivisible by three.

    Worth pinning rather than treating as coincidence. It means the "who gets the
    odd coin" question never arises in the real game, and if a future payout row
    breaks the pattern that is a strong hint the number was transcribed wrong.
    """
    payers = real_rules.play.players - 1
    amounts = []
    for mono in (False, True):
        amounts.append(real_rules.payout(HandKind.TRIPLE, monochrome=mono))
        for members in real_rules.cards.group_members:
            amounts.append(
                real_rules.payout(HandKind.GROUP, group_size=len(members), monochrome=mono)
            )
    amounts.append(real_rules.bonus_per_copy)

    offenders = [a for a in amounts if a % payers]
    assert not offenders, f"these do not divide by {payers}: {sorted(set(offenders))}"


def test_real_rules_file_loads(real_rules):
    """The live config must always be loadable — it is edited by hand."""
    assert real_rules.play.hand_limit == 7
    assert real_rules.play.players == 4
    assert real_rules.deck_size == 100
    assert real_rules.cards.n_groups == 4
    assert 14 <= real_rules.cards.n_chars <= 19
    assert len(real_rules.rules_hash) == 64


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda c: c["deck"].__setitem__("size", 100_000), "exceeds capacity"),
        # 8 still fits the deck, so this isolates the hand-limit check rather than
        # tripping the capacity one first.
        (lambda c: c["play"].__setitem__("deal_size", 8), "hand_limit"),
        (lambda c: c["groups"].__setitem__(0, {"id": "g0", "members": []}), "no members"),
        (lambda c: c["groups"][0]["members"].append("nope"), "not in the roster"),
    ],
)
def test_invalid_configs_are_rejected_with_a_useful_message(mutate, expected):
    cfg = build_config(16, [5, 4, 3, 4])
    mutate(cfg)
    with pytest.raises(ValueError, match=expected):
        Rules.from_dict(cfg)
