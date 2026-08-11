"""Properties of rebuilding a game from a public view plus a hypothesis.

The determinizer is the piece that decides whether search can ever run against the
real game. Everything it produces has to come from `PublicState` and a belief
particle, and the result has to be a state the engine will accept and play
forward correctly. Both halves are tested here, and the first one is tested by the
sharpest check available: hand it the *true* world and require that it reconstructs
the real game exactly.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import HealthCheck, given, settings

from pokajan.agents.base import RandomAgent
from pokajan.core.actions import DecisionType
from pokajan.core.engine import Engine
from pokajan.core.rules import Rules
from pokajan.envs.belief import Particle
from pokajan.envs.determinize import determinize
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

SLOW = settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

# Every field of GameState that a determinization must reproduce. Deck order is
# excluded on purpose — it is the thing being hypothesised — and is checked as a
# multiset instead.
MIRRORED = (
    "hands", "coins", "table", "scored", "discards", "recent_discards",
    "calls_made", "coins_won", "coins_paid", "current_seat", "turn_index",
    "phase", "coins_minted", "last_discard_slot", "last_discard_seat",
    "chain_seat", "chain_needs_discard", "claim_eligible",
)


def true_particle(engine: Engine) -> Particle:
    """The actual hidden state, dressed up as a belief particle."""
    space = engine.state.rules.cards
    deck = [0] * space.n_slots
    for slot in engine.state.deck:
        deck[slot] += 1
    composition = [
        deck[i]
        + sum(hand[i] for hand in engine.state.hands)
        + engine.state.table[i]
        + engine.state.scored[i]
        for i in range(space.n_slots)
    ]
    return Particle(
        hands=[hand[:] for hand in engine.state.hands],
        deck=deck,
        composition=composition,
        weight=1.0,
    )


def walk(rules: Rules, seed: int, steps: int):
    """Yield `(engine, seat, decision, slot)` at each decision point of a game."""
    engine = Engine.new_game(rules, seed=seed)
    agents = [RandomAgent(seed=seed * 4 + i) for i in range(rules.play.players)]
    for _ in range(steps):
        if engine.finished:
            break
        pending = engine.pending_seats()
        if not pending:
            break
        yield (engine, *pending[0])
        engine.submit([agents[r.seat].act(r) for r in engine.pending_decisions()])


@given(rules_configs())
@SLOW
def test_the_true_world_reconstructs_the_real_game(config):
    """Given the actual hidden cards, the rebuild must be indistinguishable.

    This is the test that makes searching from a `PublicState` trustworthy. If a
    determinization drifts from the real state in any field, every rollout run from
    it measures a game that is not the one being played — and the symptom would be
    a search that quietly recommends the wrong move, never an error.
    """
    rules = Rules.from_dict(config)
    checked = 0

    for engine, seat, decision, slot in walk(rules, seed=31, steps=25):
        rebuilt = determinize(
            rules,
            engine.public_state(seat),
            true_particle(engine),
            seat=seat,
            decision=decision,
            claimable_slot=slot,
            chain_from_claim=engine.state.chain_from_claim,
            rng=random.Random(0),
        )
        for field in MIRRORED:
            assert getattr(rebuilt.state, field) == getattr(engine.state, field), (
                f"{decision.name}: {field} differs"
            )
        assert sorted(rebuilt.state.deck) == sorted(engine.state.deck)
        checked += 1

    assert checked > 0


@given(rules_configs())
@SLOW
def test_determinized_games_are_playable_and_conserve_cards(config):
    """A rebuilt world must be a legal game, not just a plausible-looking one."""
    from pokajan.agents.rollout import RolloutPolicy, play_out

    rules = Rules.from_dict(config)
    policy = RolloutPolicy(rules, seed=1)
    played = 0

    for engine, seat, decision, slot in walk(rules, seed=32, steps=12):
        rebuilt = determinize(
            rules,
            engine.public_state(seat),
            true_particle(engine),
            seat=seat,
            decision=decision,
            claimable_slot=slot,
            chain_from_claim=engine.state.chain_from_claim,
            rng=random.Random(played),
        )
        assert rebuilt.state.cards_in_play() == rules.deck_size
        play_out(rebuilt, policy)
        assert rebuilt.state.cards_in_play() == rules.deck_size
        assert all(c >= rules.end.coin_floor for c in rebuilt.state.coins)
        played += 1

    assert played > 0


@given(rules_configs())
@SLOW
def test_the_searching_seat_is_always_offered_the_claim(config):
    """It was asked, so it is eligible — whatever the hypothesis happens to say.

    Eligibility for everyone else is derived from the hypothesised hands, which is
    the point: the search gets to discover that a rival could have taken the card
    in this world and not in that one. But deriving it for the searching seat too
    would occasionally drop it out of its own claim window and leave the search
    unable to evaluate the move it was asked about.
    """
    rules = Rules.from_dict(config)
    checked = 0

    for engine, seat, decision, slot in walk(rules, seed=33, steps=60):
        if decision is not DecisionType.CLAIM:
            continue
        # A hypothesis with an empty hand for every opponent: nobody else is
        # eligible under it, so only the guarantee keeps the seat in the window.
        space = rules.cards
        blank = Particle(
            hands=[
                engine.state.hands[seat][:] if s == seat else space.zeros()
                for s in range(rules.play.players)
            ],
            deck=space.zeros(),
            composition=space.zeros(),
            weight=1.0,
        )
        rebuilt = determinize(
            rules,
            engine.public_state(seat),
            blank,
            seat=seat,
            decision=decision,
            claimable_slot=slot,
            rng=random.Random(0),
        )
        assert seat in rebuilt.state.claim_eligible
        assert seat in {s for s, _, _ in rebuilt.pending_seats()}
        checked += 1

    if checked == 0:
        pytest.skip("no claim window arose in this config")
