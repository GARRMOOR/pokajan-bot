"""Properties the evaluation harness must have to be worth trusting.

Every later decision in this project — is the PIMC agent better, did PPO improve,
which checkpoint ships — is read off this harness. A biased or noisy harness would
not announce itself; it would just quietly approve the wrong agent. So it gets
tested like production code rather than like a script.
"""

from __future__ import annotations

import pytest

from pokajan.core.engine import Engine
from pokajan.train.evaluate import AGENTS, duplicate_match, play_rotation

pytestmark = pytest.mark.invariant


def test_every_rotation_of_a_seed_plays_the_same_deck(fixture_rules):
    """The premise of duplicate dealing.

    If the deck moved between rotations, the test agent and the baseline would be
    facing different games and the whole variance reduction would be imaginary.
    """
    decks = [Engine.new_game(fixture_rules, seed=77).state.deck for _ in range(4)]
    bonuses = [Engine.new_game(fixture_rules, seed=77).state.bonus_character for _ in range(4)]

    assert all(deck == decks[0] for deck in decks)
    assert all(bonus == bonuses[0] for bonus in bonuses)
    # ...and a different seed must actually deal differently, or the harness is
    # measuring one game over and over.
    assert Engine.new_game(fixture_rules, seed=78).state.deck != decks[0]


def test_the_test_agent_occupies_each_seat_in_turn(fixture_rules):
    for rotation in range(fixture_rules.play.players):
        row = play_rotation(fixture_rules, "greedy", "random", seed=5, rotation=rotation)
        assert row["rotation"] == rotation
        assert row["seed"] == 5


def test_reports_are_reproducible(fixture_rules):
    """Two runs of the same matchup must agree exactly.

    Not a nicety: a comparison between two agents is only meaningful if rerunning
    it does not move the number, and any hidden global RNG would break that.
    """
    first = duplicate_match(fixture_rules, "greedy", "random", seeds=6, workers=1)
    second = duplicate_match(fixture_rules, "greedy", "random", seeds=6, workers=1)

    assert first.mean_delta == second.mean_delta
    assert first.stderr == second.stderr
    assert first.win_rate == second.win_rate
    assert first.mean_by_seat == second.mean_by_seat


def test_report_bookkeeping(fixture_rules):
    players = fixture_rules.play.players
    report = duplicate_match(fixture_rules, "heuristic-fast", "random", seeds=5, workers=1)

    assert report.seeds == 5
    assert report.games == 5 * players
    assert len(report.mean_by_seat) == players
    assert report.rules_hash == fixture_rules.rules_hash

    low, high = report.ci95
    assert low <= report.mean_delta <= high
    assert 0.0 <= report.win_rate <= 1.0
    assert 0.0 <= report.p_above_start <= 1.0
    assert 0.0 <= report.bust_rate <= 1.0
    assert sum(report.end_reasons.values()) == report.games


def test_a_mirror_match_shows_no_edge(fixture_rules):
    """The same policy on both sides must come out level.

    This is the harness's own null hypothesis. A seat-order bias that duplicate
    dealing failed to cancel, or an agent seeded so that one chair plays
    differently, would show up here as a significant edge between an agent and
    itself. Fully deterministic, so this either holds or it does not — there is
    nothing to flake.
    """
    report = duplicate_match(fixture_rules, "random", "random", seeds=60, workers=1)

    low, high = report.ci95
    assert low <= 0.0 <= high, (
        f"an agent beat itself by {report.mean_delta:+.1f} coins — the harness is biased"
    )


def test_registered_agents_all_build(fixture_rules):
    for name in AGENTS:
        agent = AGENTS[name](fixture_rules, 0)
        assert hasattr(agent, "act")
        assert isinstance(agent.name, str)
