"""Properties every game must satisfy, under any rules config and any play.

These are the tests that matter most right now. Almost everything an agent could
exploit shows up here as a broken invariant: cards appearing from nowhere, coins
minted when nobody went bankrupt, a game that never ends. A reinforcement learner
finds bugs like these far faster than a human reading the code does, and it will
happily build its whole strategy on one.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pokajan.agents.base import GreedyCallerAgent, RandomAgent
from pokajan.core.engine import Engine
from pokajan.core.rules import Rules
from pokajan.core.state import EndReason, Phase
from pokajan.envs.driver import play_game
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def seats(rules: Rules, seed: int, greedy: bool = False):
    if greedy:
        return [GreedyCallerAgent(rules, seed=seed + i) for i in range(rules.play.players)]
    return [RandomAgent(seed=seed + i) for i in range(rules.play.players)]


@given(cfg=rules_configs(), seed=st.integers(0, 10_000), greedy=st.booleans())
@SETTINGS
def test_cards_are_conserved_throughout(cfg, seed, greedy):
    """Nothing is created or destroyed — cards only move between piles."""
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed)

    decisions = 0
    while not engine.finished:
        assert engine.state.cards_in_play() == rules.deck_size, (
            f"card count drifted at turn {engine.state.turn_index}"
        )
        pending = engine.pending_decisions()
        if not pending:
            break
        agents = seats(rules, seed, greedy)
        engine.submit([agents[r.seat].act(r) for r in pending])
        decisions += 1
        assert decisions < 20_000, "game is not progressing"

    assert engine.state.cards_in_play() == rules.deck_size


@given(cfg=rules_configs(), seed=st.integers(0, 10_000), greedy=st.booleans())
@SETTINGS
def test_coin_accounting_including_the_bankruptcy_mint(cfg, seed, greedy):
    """Coins are conserved *except* for what the bankruptcy floor mints.

    Written as `total == start + minted` rather than `total == start` on purpose.
    The real rule is that a payer who cannot cover their share stops at zero while
    the caller still receives the full amount, so the naive conservation check
    would be wrong — and quietly "fixing" it by deleting the check would lose the
    only guard on the whole payment path.
    """
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed)
    start = rules.play.initial_coins * rules.play.players
    agents = seats(rules, seed, greedy)

    while not engine.finished:
        s = engine.state
        assert sum(s.coins) == start + s.coins_minted
        assert all(c >= 0 for c in s.coins), "a seat went below zero"
        pending = engine.pending_decisions()
        if not pending:
            break
        engine.submit([agents[r.seat].act(r) for r in pending])

    s = engine.state
    assert sum(s.coins) == start + s.coins_minted
    assert s.coins_minted >= 0


@given(cfg=rules_configs(), seed=st.integers(0, 10_000), greedy=st.booleans())
@SETTINGS
def test_every_game_terminates_for_a_declared_reason(cfg, seed, greedy):
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed)
    result = play_game(engine, seats(rules, seed, greedy))

    assert engine.finished
    assert result.end_reason in {EndReason.DECK_EMPTY.value, EndReason.BANKRUPT.value}
    assert engine.state.phase is Phase.FINISHED

    # And the reason must match the state it left behind.
    if result.end_reason == EndReason.BANKRUPT.value:
        assert min(result.final_coins) <= rules.end.coin_floor
    else:
        assert engine.state.deck_remaining == 0


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_hand_sizes_stay_within_bounds(cfg, seed):
    """A hand may exceed the limit only between drawing and discarding."""
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed)
    limit = rules.play.hand_limit
    agents = seats(rules, seed)

    while not engine.finished:
        s = engine.state
        for seat in range(s.players):
            size = s.hand_size(seat)
            assert size <= limit + 1, f"seat {seat} holds {size}, limit is {limit}"
        pending = engine.pending_decisions()
        if not pending:
            break
        engine.submit([agents[r.seat].act(r) for r in pending])


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_only_legal_actions_are_ever_offered(cfg, seed):
    """Every request must offer at least one legal action, and discards must be held."""
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed)
    n_slots = rules.cards.n_slots
    agents = seats(rules, seed)

    while not engine.finished:
        pending = engine.pending_decisions()
        if not pending:
            break
        for request in pending:
            assert any(request.legal_mask), f"seat {request.seat} has no legal action"
            hand = engine.state.hands[request.seat]
            for slot in range(n_slots):
                if request.legal_mask[slot]:
                    assert hand[slot] > 0, "offered a discard of a card not held"
        engine.submit([agents[r.seat].act(r) for r in pending])


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_public_state_never_leaks_another_hand(cfg, seed):
    """The screen reader will never see opponents' hands, so neither may training.

    Only hand *sizes* are public. If a hidden count ever reached the observation,
    the agent would learn a read that vanishes the moment it plays the real game.
    """
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed)
    agents = seats(rules, seed)

    while not engine.finished:
        pending = engine.pending_decisions()
        if not pending:
            break
        for request in pending:
            view = request.state
            assert view.viewer == request.seat
            assert view.hand == engine.state.hands[request.seat]
            assert not hasattr(view, "hands")
            for seat_view in view.seats:
                assert isinstance(seat_view.hand_size, int)
                assert not hasattr(seat_view, "hand")
        engine.submit([agents[r.seat].act(r) for r in pending])


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_claim_windows_ask_everyone_against_the_same_state(cfg, seed):
    """No seat's claim decision may depend on another's.

    Every request in a claim window must carry an identical public picture. If
    these ever diverged it would mean the engine had started resolving claims
    mid-window, leaking "the seat before you already passed" into later seats.
    """
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed)
    agents = seats(rules, seed)
    saw_multi_claim = False

    while not engine.finished:
        pending = engine.pending_decisions()
        if not pending:
            break
        if len(pending) > 1:
            saw_multi_claim = True
            assert all(r.decision == "CLAIM" for r in pending)
            first = pending[0].state
            for other in pending[1:]:
                assert other.state.table == first.table
                assert other.state.scored == first.scored
                assert other.state.deck_remaining == first.deck_remaining
                assert other.state.last_discard_slot == first.last_discard_slot
                assert [s.coins for s in other.state.seats] == [
                    s.coins for s in first.seats
                ]
        engine.submit([agents[r.seat].act(r) for r in pending])

    # Not asserted every run — a short game may contain no contested discard.
    _ = saw_multi_claim


@given(cfg=rules_configs(), seed=st.integers(0, 10_000))
@SETTINGS
def test_clone_is_independent_and_faithful(cfg, seed):
    """Search forks the state mid-game, so a clone must diverge cleanly."""
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed)
    agents = seats(rules, seed)

    for _ in range(12):
        if engine.finished:
            break
        pending = engine.pending_decisions()
        if not pending:
            break
        engine.submit([agents[r.seat].act(r) for r in pending])

    if engine.finished:
        return

    fork = engine.clone()
    assert fork.state.coins == engine.state.coins
    assert fork.state.hands == engine.state.hands
    assert fork.state.deck == engine.state.deck
    assert [r.seat for r in fork.pending_decisions()] == [
        r.seat for r in engine.pending_decisions()
    ]

    before = [h[:] for h in engine.state.hands]
    play_game(fork, [RandomAgent(seed=seed + 99 + i) for i in range(rules.play.players)])
    assert engine.state.hands == before, "playing the clone mutated the original"
    assert not engine.finished
