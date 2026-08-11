"""Cases that pin what the belief is actually *for*.

The invariant tests prove the belief conserves cards. These prove it knows
something — specifically, the two things a human at the real table cannot work
out: how many of each card the deck was built with, and which cards the other
players have already shown they cannot use.

Unlike the payout scenarios, these pin *properties* rather than numbers. The exact
error of a posterior depends on the deck and the seed; what must hold is that it
beats the alternative, and by a wide margin rather than a hair.
"""

from __future__ import annotations

import pytest

from pokajan.agents.base import RandomAgent
from pokajan.core.cards import Counts
from pokajan.core.engine import Engine
from pokajan.core.evaluate import can_call_using
from pokajan.core.rules import Rules
from pokajan.envs.belief import Belief
from pokajan.server.protocol import PublicState, SeatView

pytestmark = pytest.mark.scenario


def true_composition(engine: Engine) -> Counts:
    """What the deck was actually built with — never visible to a player."""
    state = engine.state
    counts = [0] * state.rules.cards.n_slots
    for slot in state.deck:
        counts[slot] += 1
    for hand in state.hands:
        for slot, n in enumerate(hand):
            counts[slot] += n
    for slot, n in enumerate(state.table):
        counts[slot] += n
    for slot, n in enumerate(state.scored):
        counts[slot] += n
    return counts


def run(rules: Rules, seed: int, steps: int, viewer: int = 0):
    """Play a while, feeding every view to a belief as an agent would see it."""
    engine = Engine.new_game(rules, seed=seed)
    agents = [RandomAgent(seed=seed * 7 + i) for i in range(rules.play.players)]
    belief = Belief(
        rules, viewer, seed=seed, bonus_character=engine.state.bonus_character
    )
    belief.observe(engine.public_state(viewer))

    for _ in range(steps):
        if engine.finished:
            break
        pending = engine.pending_decisions()
        if not pending:
            break
        engine.submit([agents[r.seat].act(r) for r in pending])
        belief.observe(engine.public_state(viewer))

    return engine, belief


def mean_absolute_error(estimate, truth) -> float:
    return sum(abs(a - b) for a, b in zip(estimate, truth)) / len(truth)


@pytest.mark.parametrize("rules_name", ["fixture", "real"])
def test_posterior_beats_the_in_game_counter(rules_name, fixture_rules, real_rules):
    """The headline claim of the whole project, as a test.

    The real game shows a "remaining cards" list built from every card the player
    has not seen, drawn from the full 9-copies-per-holomem pool — 153 cards for a
    17-holomem lineup, when the deck holds 100. `naive_unseen` reproduces exactly
    that reasoning. `expected_unseen` is what inference from observed play gives
    instead.

    If this ever fails, the composition posterior has stopped earning its place and
    the agent would be better off with the counter a human reads.
    """
    rules = fixture_rules if rules_name == "fixture" else real_rules

    posterior_error, naive_error = [], []
    for seed in range(6):
        engine, belief = run(rules, seed=seed, steps=40)
        truth = true_composition(engine)
        actual_unseen = [truth[s] - belief.seen[s] for s in range(rules.cards.n_slots)]

        posterior_error.append(mean_absolute_error(belief.expected_unseen(), actual_unseen))
        naive_error.append(mean_absolute_error(belief.naive_unseen(), actual_unseen))

    posterior = sum(posterior_error) / len(posterior_error)
    naive = sum(naive_error) / len(naive_error)

    assert posterior < naive, (
        f"posterior {posterior:.3f} did not beat the in-game counter {naive:.3f}"
    )
    # A margin, not a coin flip. The counter is wrong structurally, not by luck.
    assert posterior < naive * 0.75


def test_seeing_one_card_lowers_the_estimate_of_the_others(fixture_rules):
    """The deck total couples every slot to every other, and humans cannot use it.

    Because the deck holds exactly `deck_size` cards, learning that one card is
    plentiful is *also* evidence that others are thin. The in-game counter cannot
    express this at all: it treats every unseen card as independently available.
    """
    rules = fixture_rules
    n = rules.cards.n_slots
    cap = rules.cards.max_per_color

    def belief_seeing(table: Counts) -> Belief:
        belief = Belief(rules, 0, seed=3)
        belief.observe(public_state_with(rules, table))
        return belief

    blank = belief_seeing([0] * n)
    loaded_table = [0] * n
    loaded_table[0] = cap                      # every copy of one card, on the table
    loaded = belief_seeing(loaded_table)

    assert loaded.expected_composition()[0] > blank.expected_composition()[0]

    others_blank = sum(blank.expected_composition()[1:])
    others_loaded = sum(loaded.expected_composition()[1:])
    assert others_loaded < others_blank


def public_state_with(rules: Rules, table: Counts) -> PublicState:
    """A minimal state with a given table and empty hands, for controlled tests."""
    n = rules.cards.n_slots
    players = rules.play.players
    return PublicState(
        game_id="synthetic",
        turn_index=0,
        viewer=0,
        hand=[0] * n,
        seats=[
            SeatView(
                seat=s,
                coins=rules.play.initial_coins,
                hand_size=0,
                discards=[0] * n,
                calls_made=0,
                coins_won=0,
                coins_paid=0,
            )
            for s in range(players)
        ],
        deck_remaining=rules.deck_size - sum(table),
        table=list(table),
        scored=[0] * n,
        last_discard_slot=None,
        last_discard_seat=None,
        current_seat=0,
        phase="AWAIT_DISCARD",
        bonus_character=rules.bonus_character,
        recent_discards=[[] for _ in range(players)],
    )


def test_a_card_nobody_claimed_is_evidence_nobody_could(real_rules):
    """Passing on a discard is free information, and the belief must spend it.

    The engine only offers a claim to seats that could legally make one, so a card
    that stays on the table has been declined by every seat that could have used
    it. That rules out a specific shape of hand for each of them — the sharpest
    read the rules hand out, and one almost no human tracks.

    Both beliefs below are seeded identically and observe identical states, so they
    sample the *same* hypothetical worlds. The only difference is whether the pass
    evidence is allowed to weigh them, which isolates the mechanism exactly.
    """
    engine, informed = run(real_rules, seed=4, steps=60)
    assert informed.pass_events, "no discard went unclaimed — scenario is not exercising this"

    # Same seed, same observations, evidence discarded.
    _, ignorant = run(real_rules, seed=4, steps=60)
    ignorant.pass_events = []

    slot = informed.pass_events[-1].slot
    discarder = informed.pass_events[-1].discarder

    def probability_someone_could_have_claimed(belief: Belief) -> float:
        belief.rng.seed(99)
        total = 0.0
        for particle in belief.sample(256):
            for seat in range(real_rules.play.players):
                if seat in (belief.viewer, discarder):
                    continue
                probe = particle.hands[seat][:]
                probe[slot] += 1
                if can_call_using(
                    real_rules, probe, slot, bonus_character=engine.state.bonus_character
                ):
                    total += particle.weight
                    break
        return total

    informed_p = probability_someone_could_have_claimed(informed)
    ignorant_p = probability_someone_could_have_claimed(ignorant)

    assert informed_p < ignorant_p, (
        f"pass evidence did not reduce the estimate ({informed_p:.4f} vs {ignorant_p:.4f})"
    )
