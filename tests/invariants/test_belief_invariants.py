"""Properties the belief must hold whatever the rules turn out to be.

The deck composition rule is the one thing about this game we genuinely do not
know, so these run against randomly generated configs and assert only what is true
of *any* of them. What they mostly guard is the accounting: a belief that quietly
loses or invents cards would still look plausible in a heatmap and would poison
every probability the agent computes from it.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from pokajan.agents.base import RandomAgent
from pokajan.core.engine import Engine
from pokajan.core.rules import Rules
from pokajan.envs.belief import Belief
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

# The prior is estimated by sampling build_deck, and these tests build a fresh one
# per config rather than reusing a cached hash, so keep it small.
PRIOR_SAMPLES = 150

SLOW = settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def play_a_while(rules: Rules, seed: int, steps: int) -> Engine:
    """Advance a game far enough that there is something to infer from."""
    engine = Engine.new_game(rules, seed=seed)
    agents = [RandomAgent(seed=seed + i) for i in range(rules.play.players)]
    for _ in range(steps):
        if engine.finished:
            break
        pending = engine.pending_decisions()
        if not pending:
            break
        engine.submit([agents[r.seat].act(r) for r in pending])
    return engine


def belief_for(engine: Engine, seat: int, *, seed: int = 0) -> Belief:
    belief = Belief(
        engine.state.rules,
        seat,
        seed=seed,
        bonus_character=engine.state.bonus_character,
        prior_samples=PRIOR_SAMPLES,
    )
    belief.observe(engine.public_state(seat))
    return belief


@given(rules_configs())
@SLOW
def test_composition_totals_the_deck(config):
    """The deck holds exactly `deck_size` cards, and the posterior must agree.

    Per-slot updates are independent, so this only holds because `_tilt` couples
    them back together afterwards. Without it the total drifts by several cards
    and every derived probability is scaled wrong.
    """
    rules = Rules.from_dict(config)
    engine = play_a_while(rules, seed=11, steps=30)
    belief = belief_for(engine, 0)

    assert sum(belief.expected_composition()) == pytest.approx(rules.deck_size, abs=1e-6)


@given(rules_configs())
@SLOW
def test_never_believes_fewer_cards_than_it_has_seen(config):
    rules = Rules.from_dict(config)
    engine = play_a_while(rules, seed=12, steps=40)
    belief = belief_for(engine, 1)

    posterior = belief.composition_posterior()
    for slot, seen in enumerate(belief.seen):
        assert sum(posterior[slot][:seen]) == pytest.approx(0.0, abs=1e-12)
        assert belief.expected_composition()[slot] >= seen - 1e-9


@given(rules_configs())
@SLOW
def test_unseen_accounts_for_exactly_the_hidden_cards(config):
    """Expected unseen copies must total the cards this seat cannot see.

    This is the number the in-game counter gets wrong, and it is wrong by a wide
    margin: it counts the whole theoretical pool rather than the deck.
    """
    rules = Rules.from_dict(config)
    engine = play_a_while(rules, seed=13, steps=50)
    belief = belief_for(engine, 2)

    assert sum(belief.expected_unseen()) == pytest.approx(belief.hidden_total(), abs=1e-6)


@given(rules_configs())
@SLOW
def test_particles_are_legal_worlds(config):
    """Every sampled world must be one the game could actually be in."""
    rules = Rules.from_dict(config)
    engine = play_a_while(rules, seed=14, steps=45)
    state = engine.public_state(0)
    belief = belief_for(engine, 0)

    particles = belief.sample(16)
    assert len(particles) == 16
    assert sum(p.weight for p in particles) == pytest.approx(1.0)
    assert all(p.weight > 0.0 for p in particles)

    sizes = {v.seat: v.hand_size for v in state.seats}
    for p in particles:
        assert sum(p.composition) == rules.deck_size
        assert all(0 <= c <= rules.cards.max_per_color for c in p.composition)
        # A world may not contain fewer copies than have already been revealed.
        assert all(c >= s for c, s in zip(p.composition, belief.seen))
        # Hands and deck must partition exactly the cards not already visible.
        for seat, hand in enumerate(p.hands):
            assert sum(hand) == sizes[seat]
        assert p.hands[0] == state.hand
        assert sum(p.deck) == state.deck_remaining
        for slot in range(rules.cards.n_slots):
            # Card conservation inside one hypothesised world: everything the deck
            # was built with is now in a hand, on the table, or spent on a call.
            held = sum(hand[slot] for hand in p.hands)
            assert held + p.deck[slot] + state.table[slot] + state.scored[slot] == (
                p.composition[slot]
            )


@given(rules_configs())
@SLOW
def test_location_marginals_conserve_cards(config):
    rules = Rules.from_dict(config)
    engine = play_a_while(rules, seed=15, steps=35)
    state = engine.public_state(3)
    belief = belief_for(engine, 3)

    rows = belief.location(belief.sample(16))
    assert len(rows) == rules.play.players + 1
    for seat, view in enumerate(state.seats):
        assert sum(rows[seat]) == pytest.approx(view.hand_size, abs=1e-6)
    assert sum(rows[-1]) == pytest.approx(state.deck_remaining, abs=1e-6)


@given(rules_configs())
@SLOW
def test_rejects_another_seats_view(config):
    """A belief must never be fed a view it is not entitled to.

    Cheap to get wrong when wiring four agents up, and the symptom would be an
    agent that plays suspiciously well rather than an error.
    """
    rules = Rules.from_dict(config)
    engine = play_a_while(rules, seed=16, steps=10)
    belief = Belief(rules, 0, seed=1, prior_samples=PRIOR_SAMPLES)

    with pytest.raises(ValueError):
        belief.observe(engine.public_state(1))
