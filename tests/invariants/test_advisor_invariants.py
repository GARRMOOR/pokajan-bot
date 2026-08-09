"""Properties of the advice, as distinct from the play.

The advisor is the only producer of `Recommendation`, and both M7 surfaces consume
it — the browser hint panel and the overlay pinned over the real game. Its failure
mode is not a crash but a plausible sentence that is wrong, so what gets asserted
here is the shape of the claim: a legal action, a confidence that means what it
says, and text that renders wherever it is sent.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from pokajan.agents.advisor import Advisor
from pokajan.agents.base import RandomAgent
from pokajan.core.engine import Engine
from pokajan.core.rules import Rules
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

FEW = settings(
    max_examples=6,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@given(rules_configs())
@FEW
def test_advice_is_always_playable(config):
    """Advice a human cannot follow is worse than none."""
    rules = Rules.from_dict(config)
    engine = Engine.new_game(rules, seed=71)
    others = [RandomAgent(seed=i) for i in range(rules.play.players)]
    advisor = Advisor(rules, seed=0, particles=6)

    checked = 0
    while not engine.finished and checked < 25:
        pending = engine.pending_decisions()
        if not pending:
            break
        for request in pending:
            if request.seat != 0:
                continue
            rec = advisor.recommend(request)

            assert request.legal_mask[rec.action], "recommended an illegal move"
            assert 0.0 <= rec.confidence <= 1.0
            assert rec.action_label
            assert rec.reasoning
            # Rendered into a browser panel, a transparent overlay and sometimes a
            # terminal; a dash that survives two of the three is a bug.
            assert rec.reasoning.isascii()
            for alt in rec.alternatives:
                assert alt["action"] != rec.action
                assert request.legal_mask[alt["action"]]
                # The rendered gap. Never negative, or the "recommendation" is
                # not the best option the advisor found.
                assert alt["behind"] >= -1e-9, "an alternative scored above the advice"
            checked += 1
        engine.submit([others[r.seat].act(r) for r in pending])

    assert checked > 0


def test_confidence_rises_with_the_margin(fixture_rules):
    """It is a probability, not a flourish.

    Grounded in the measured spread of the danger estimate between independent
    draws of the belief, so a wider margin has to read as more certain and a zero
    margin as a coin flip. If this ever went flat, every hint would look equally
    trustworthy — including the ones that are not.
    """
    advisor = Advisor(fixture_rules, seed=1, particles=48)

    assert advisor._confidence(0.0) == pytest.approx(0.5)
    margins = [advisor._confidence(gap) for gap in (0.0, 5.0, 20.0, 100.0)]
    assert margins == sorted(margins)
    assert margins[-1] > 0.99


def test_more_particles_makes_the_same_margin_more_certain(fixture_rules):
    """Because the noise it is measured against shrinks as 1/sqrt(n)."""
    coarse = Advisor(fixture_rules, seed=1, particles=48)
    fine = Advisor(fixture_rules, seed=1, particles=1024)

    assert fine._confidence(10.0) > coarse._confidence(10.0)


def test_a_toss_up_is_described_as_one(fixture_rules):
    """The reasoning must not invent a rationale for noise.

    An explanation that names the deciding consideration reads as authoritative.
    When the margin is inside the sampling error there is no deciding
    consideration, and saying so is the honest output.
    """
    advisor = Advisor(fixture_rules, seed=1, particles=48)
    best = {"label": "Alpha (blue)", "score": 100.0, "hand_value": 100.0, "danger": 0.0}
    close = {"label": "Bravo (blue)", "score": 99.0, "hand_value": 110.0, "danger": 11.0}
    clear = {"label": "Bravo (blue)", "score": 10.0, "hand_value": 20.0, "danger": 10.0}

    toss_up = advisor._explain_discard(best, close, advisor._confidence(1.0))
    decided = advisor._explain_discard(best, clear, advisor._confidence(90.0))

    assert "coin flip" in toss_up
    assert "Mainly" not in toss_up
    assert "Mainly" in decided
