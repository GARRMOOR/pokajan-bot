"""Properties of the observation encoding, under any roster.

The encoding is one of the few things in this project that is genuinely expensive
to change: every trained checkpoint is tied to it. So these tests are less about
catching bugs today than about making a silent layout change impossible — a block
that shifts by one would train perfectly happily and mean nothing.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings

from pokajan.core.actions import DecisionType
from pokajan.core.engine import Engine
from pokajan.core.rules import Rules
from pokajan.envs.belief import Belief
from pokajan.envs.obs import ObsSpec, encode
from tests.conftest import build_config, rules_configs

pytestmark = pytest.mark.invariant

SLOW = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@given(rules_configs())
@SLOW
def test_blocks_tile_the_vector_exactly(config):
    """No gaps, no overlaps, no slack at the end."""
    spec = ObsSpec.build(Rules.from_dict(config))

    cursor = 0
    for name, offset, size in spec.layout():
        assert offset == cursor, f"block {name} is not contiguous"
        assert size > 0
        cursor += size
    assert cursor == spec.size


@given(rules_configs())
@SLOW
def test_encoding_is_finite_and_scaled(config):
    rules = Rules.from_dict(config)
    spec = ObsSpec.build(rules)
    engine = Engine.new_game(rules, seed=5)
    state = engine.public_state(0)

    belief = Belief(rules, 0, seed=1, bonus_character=engine.state.bonus_character,
                    prior_samples=150)
    belief.observe(state)

    vector = encode(
        spec,
        state,
        belief=belief,
        particles=belief.sample(8),
        decision=DecisionType.DISCARD,
        bonus_character=engine.state.bonus_character,
    )

    assert len(vector) == spec.size
    assert all(math.isfinite(v) for v in vector)
    # Not a strict [0,1] claim — a few features legitimately exceed it — but a
    # network fed values in the hundreds trains badly, and that is what a missed
    # normalisation looks like.
    assert all(-2.0 <= v <= 5.0 for v in vector)


@given(rules_configs())
@SLOW
def test_belief_blocks_are_zero_without_a_belief(config):
    """The encoder must work uninformed, and say so honestly rather than guess."""
    rules = Rules.from_dict(config)
    spec = ObsSpec.build(rules)
    engine = Engine.new_game(rules, seed=6)
    vector = encode(spec, engine.public_state(0))

    for name in ("composition_mean", "composition_std", "unseen_mean", "deck_mean",
                 "opponent_hands"):
        assert not any(spec.view(vector, name))


def test_opponents_are_encoded_relative_to_the_viewer(fixture_rules):
    """Seat 1 must look identical to seat 0 as "the player on my left".

    Nothing in Pokajan depends on which chair you occupy, so absolute seat indices
    in the observation would make the network learn one strategy four times. This
    is the test that pins the rotation.
    """
    spec = ObsSpec.build(fixture_rules)
    n = fixture_rules.cards.n_slots
    engine = Engine.new_game(fixture_rules, seed=9)

    # Give each seat a distinctive discard history so the rotation is visible.
    for seat in range(fixture_rules.play.players):
        engine.state.discards[seat][seat] += 1

    for viewer in range(fixture_rules.play.players):
        vector = encode(spec, engine.public_state(viewer))
        block = spec.view(vector, "opponent_discards")
        for i in range(fixture_rules.play.players - 1):
            expected_seat = (viewer + 1 + i) % fixture_rules.play.players
            slice_ = block[i * n : (i + 1) * n]
            assert slice_[expected_seat] > 0, (
                f"viewer {viewer} expected seat {expected_seat} at opponent index {i}"
            )


def test_signature_tracks_the_roster():
    """A checkpoint is only valid for the layout it was trained under."""
    small = ObsSpec.build(Rules.from_dict(build_config(14, [4, 4, 3, 3])))
    large = ObsSpec.build(Rules.from_dict(build_config(19, [5, 5, 5, 4])))

    assert small.signature != large.signature
    assert small.size != large.size
    # Same shape, different payouts: the observation layout is unchanged, so a
    # checkpoint stays loadable and only the rules hash flags the difference.
    repriced = ObsSpec.build(
        Rules.from_dict(build_config(14, [4, 4, 3, 3], triple_payout=999))
    )
    assert repriced.signature == small.signature
