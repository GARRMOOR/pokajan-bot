"""Shared fixtures, plus the random-config generator the invariant tests use.

The generator matters more than it looks. Most of the payout table is still a
guess, so a test suite written against today's numbers would mostly be asserting
that our guesses are our guesses. Instead the invariant tests run against
*randomly generated* rules — any roster from 14 to 19 characters, any group split,
any payout scale — and assert only things that must hold however the real rules
turn out. Those tests keep their value when the numbers change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pokajan.core.rules import Rules  # noqa: E402

FIXTURE_RULES_PATH = ROOT / "tests" / "fixtures" / "rules_v1.yaml"
REAL_RULES_PATH = ROOT / "rules" / "pokajan_v1.yaml"


@pytest.fixture(scope="session")
def fixture_rules() -> Rules:
    """The frozen fixture. Golden scenarios pin to this."""
    return Rules.load(FIXTURE_RULES_PATH)


@pytest.fixture(scope="session")
def real_rules() -> Rules:
    """The live config, guesses and all."""
    return Rules.load(REAL_RULES_PATH)


def build_config(
    n_chars: int,
    group_sizes: list[int],
    *,
    hand_limit: int = 7,
    deal_size: int = 7,
    players: int = 4,
    deck_size: int = 100,
    triple_payout: int = 120,
    group_scale: int = 60,
    mono_mult: float = 3.0,
    bonus_per_copy: int = 90,
    bonus_character: str | None = None,
) -> dict:
    """A valid rules dict with the given shape."""
    chars = [{"id": f"c{i}", "name": f"Char {i}"} for i in range(n_chars)]

    groups, cursor = [], 0
    for gi, size in enumerate(group_sizes):
        members = [f"c{i}" for i in range(cursor, cursor + size)]
        groups.append({"id": f"g{gi}", "name": f"Group {gi}", "members": members})
        cursor += size

    # An entry for every group size that could exist, so validation passes for any
    # split the generator produces. `mono_mult` varies the monochrome premium
    # across generated configs precisely because the real one is not a constant —
    # anything asserting a fixed ratio should fail here.
    group_payouts = {
        size: {
            "multi": group_scale * size,
            "mono": int(group_scale * size * mono_mult),
        }
        for size in range(1, hand_limit + 1)
    }

    return {
        "version": 1,
        "name": "generated",
        "deck": {
            "size": deck_size,
            "colors": ["blue", "orange", "pink"],
            "max_per_color": 3,
            "composition": "even_as_possible",
        },
        "characters": chars,
        "groups": groups,
        "bonus_character": bonus_character,
        "play": {
            "players": players,
            "initial_coins": 1000,
            "hand_limit": hand_limit,
            "deal_size": deal_size,
            "draws_per_turn": 1,
            "refill_policy": "up_to_limit",
            "chain_allowed": True,
            "chain_is_optional": True,
            "claim_moves_turn": False,
            "discard_after_claim": False,
            "discard_after_in_turn_call": True,
            "claimable": "most_recent_discard_only",
        },
        "hands": [
            {"id": "triple", "predicate": "three_of_a_kind"},
            {"id": "group", "predicate": "full_group"},
        ],
        "payouts": {
            "table": {
                "triple": {
                    "multi": triple_payout,
                    # Triples get a much steeper monochrome premium than groups in
                    # the real game (7x vs ~3x), so the generator reproduces that
                    # asymmetry rather than a single shared multiplier.
                    "mono": int(triple_payout * mono_mult * 2),
                },
                "group": group_payouts,
            },
            "bonus": {"per_copy": bonus_per_copy, "applies_to": "scoring_set"},
            "claimed_changes_amount": False,
            "payer": {"when_claimed": "discarder", "otherwise": "split_others"},
            "caller_gains_full_amount_on_payer_bankruptcy": True,
        },
        "end_conditions": {
            "deck_empty": True,
            "coin_floor": 0,
            "floor_is_asymmetric": True,
            "multi_round": False,
        },
        "tiebreak": {"order": ["payout", "turn_order"]},
    }


@st.composite
def rules_configs(draw, min_chars: int = 14, max_chars: int = 19) -> dict:
    """Random valid configs spanning the roster sizes the real game produces.

    The real game has been observed with 14 to 19 characters across exactly four
    groups, so that is the space sampled. Group sizes are capped at the hand limit
    because a group larger than your hand could never be completed — the loader
    rejects those, and generating them would only test the error path.
    """
    n_chars = draw(st.integers(min_value=min_chars, max_value=max_chars))
    hand_limit = 7
    n_groups = 4

    # Split n_chars into 4 parts, each between 1 and hand_limit.
    while True:
        sizes = draw(
            st.lists(
                st.integers(min_value=1, max_value=hand_limit),
                min_size=n_groups,
                max_size=n_groups,
            )
        )
        if sum(sizes) == n_chars:
            break
        # Nudge toward a valid split rather than rejecting outright, which would
        # make the strategy fail its health check at these roster sizes.
        deficit = n_chars - sum(sizes)
        for i in range(n_groups):
            room = hand_limit - sizes[i] if deficit > 0 else sizes[i] - 1
            step = max(-room, min(room, deficit))
            sizes[i] += step
            deficit -= step
            if deficit == 0:
                break
        if sum(sizes) == n_chars:
            break

    bonus = draw(st.one_of(st.none(), st.integers(min_value=0, max_value=n_chars - 1)))
    return build_config(
        n_chars,
        sizes,
        hand_limit=hand_limit,
        triple_payout=draw(st.integers(min_value=10, max_value=500)),
        group_scale=draw(st.integers(min_value=5, max_value=200)),
        mono_mult=draw(st.sampled_from([1.0, 1.5, 2.0, 3.5])),
        bonus_per_copy=draw(st.integers(min_value=0, max_value=200)),
        bonus_character=None if bonus is None else f"c{bonus}",
    )
