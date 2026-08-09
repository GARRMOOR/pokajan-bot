"""The web session and the event transcript.

Two things worth guarding. The transcript is what a human will compare against a
real round, so if it disagrees with the engine's own state the whole M2 exercise
produces false conclusions. And the session must never send a seat information it
should not have — the human view goes over a socket, and it is the same
`PublicState` the screen reader will later produce.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pokajan.core.engine import Engine
from pokajan.core.rules import Rules
from pokajan.server.app import Session
from tests.conftest import rules_configs

pytestmark = pytest.mark.invariant

SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def play_out(session: Session, limit: int = 5_000) -> None:
    """Drive the human seat with the first legal action until the game ends."""
    for _ in range(limit):
        payload = session.payload()
        if payload["state"]["finished"] or payload["decision"] is None:
            return
        action = payload["decision"]["legal_mask"].index(True)
        session.act(action)
    raise AssertionError("session did not finish")


# ------------------------------------------------------------- transcript ---

@given(cfg=rules_configs(), seed=st.integers(0, 5_000))
@SETTINGS
def test_transcript_payments_reconcile_with_final_coins(cfg, seed):
    """Every coin moved must appear in the transcript, and vice versa.

    A transcript that quietly omitted a payment would be worse than no transcript:
    it would look complete while hiding exactly the arithmetic being checked.
    """
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed, record_events=True)
    from pokajan.agents.base import GreedyCallerAgent
    from pokajan.envs.driver import play_game

    play_game(engine, [GreedyCallerAgent(rules, seed=seed + s) for s in range(rules.play.players)])

    initial = rules.play.initial_coins
    reconstructed = [initial] * rules.play.players
    minted = 0
    for event in engine.state.events:
        if event["kind"] != "payment":
            continue
        reconstructed[event["to_seat"]] += event["amount"]
        for payer in event["payers"]:
            reconstructed[payer["seat"]] -= payer["paid"]
        minted += event["minted"]

    assert reconstructed == engine.final_coins()
    assert minted == engine.state.coins_minted


@given(cfg=rules_configs(), seed=st.integers(0, 5_000))
@SETTINGS
def test_transcript_pokajan_count_matches_the_engine(cfg, seed):
    rules = Rules.from_dict(cfg)
    engine = Engine.new_game(rules, seed=seed, record_events=True)
    from pokajan.agents.base import GreedyCallerAgent
    from pokajan.envs.driver import play_game

    play_game(engine, [GreedyCallerAgent(rules, seed=seed + s) for s in range(rules.play.players)])

    logged = [0] * rules.play.players
    for event in engine.state.events:
        if event["kind"] == "pokajan":
            logged[event["seat"]] += 1
    assert logged == engine.state.calls_made


@given(cfg=rules_configs(), seed=st.integers(0, 5_000))
@SETTINGS
def test_recording_events_does_not_change_play(cfg, seed):
    """The transcript must be an observer, not a participant."""
    rules = Rules.from_dict(cfg)
    from pokajan.agents.base import GreedyCallerAgent
    from pokajan.envs.driver import play_game

    results = []
    for record in (False, True):
        engine = Engine.new_game(rules, seed=seed, record_events=record)
        results.append(
            play_game(
                engine,
                [GreedyCallerAgent(rules, seed=seed + s) for s in range(rules.play.players)],
            )
        )
    assert results[0].final_coins == results[1].final_coins
    assert results[0].end_reason == results[1].end_reason
    assert results[0].calls_made == results[1].calls_made


# ---------------------------------------------------------------- session ---

@given(cfg=rules_configs(), seed=st.integers(0, 5_000), seat=st.integers(0, 3))
@SETTINGS
def test_session_plays_to_completion_and_only_ever_asks_the_human(cfg, seed, seat):
    rules = Rules.from_dict(cfg)
    session = Session(rules, human_seat=seat, seed=seed)

    steps = 0
    while True:
        payload = session.payload()
        assert payload["human_seat"] == seat
        if payload["state"]["finished"] or payload["decision"] is None:
            break
        assert payload["decision"]["seat"] == seat, "a bot seat's decision leaked to the human"
        session.act(payload["decision"]["legal_mask"].index(True))
        steps += 1
        assert steps < 5_000

    assert session.engine.finished or session.payload()["decision"] is None


@given(cfg=rules_configs(), seed=st.integers(0, 5_000))
@SETTINGS
def test_session_never_exposes_another_seats_hand(cfg, seed):
    """The payload is the same shape the screen reader will produce at M8.

    If a hidden hand reached the browser, an agent built against this view would
    depend on information that does not exist when reading the real game.
    """
    rules = Rules.from_dict(cfg)
    session = Session(rules, human_seat=0, seed=seed)
    play_out(session)

    hand_limit = rules.play.hand_limit
    for snapshot in session.history:
        state = snapshot["state"]
        assert state["viewer"] == 0
        # A hand may momentarily hold one over the limit, between draw and discard.
        assert sum(state["hand"]) <= hand_limit + 1
        for seat_view in state["seats"]:
            assert "hand" not in seat_view
            assert isinstance(seat_view["hand_size"], int)
        assert "deck" not in state
        assert isinstance(state["deck_remaining"], int)


@given(cfg=rules_configs(), seed=st.integers(0, 5_000))
@SETTINGS
def test_history_grows_monotonically_and_events_are_never_duplicated(cfg, seed):
    """The replay scrubber indexes into this, so entries must not repeat."""
    rules = Rules.from_dict(cfg)
    session = Session(rules, human_seat=0, seed=seed)
    play_out(session)

    total = sum(len(snapshot["events"]) for snapshot in session.history)
    assert total == len(session.engine.state.events)
    assert len(session.history) >= 1


def test_new_game_resets_the_transcript(real_rules):
    session = Session(real_rules, human_seat=0, seed=1)
    play_out(session)
    assert len(session.history) > 1

    session.new_game(seed=2)
    assert len(session.history) == 1
    assert session._events_sent == len(session.engine.state.events)


def test_acting_out_of_turn_is_refused(real_rules):
    session = Session(real_rules, human_seat=0, seed=3)
    play_out(session)
    with pytest.raises(ValueError, match="not your turn"):
        session.act(0)


# ------------------------------------------------------------------- hints ---
#
# The hint panel and the M8 overlay are the same code path, so what is guarded here
# is the wiring rather than the advice: that a hint is always a move the human could
# actually make, and that the belief behind it is fed by *watching*, not by being
# asked. The second one is the easy regression — moving observe() inside hint() for
# speed would leave the advisor inferring from gaps, and nothing would look broken.

def advance(session: Session, turns: int) -> None:
    """Take `turns` first-legal actions, or stop early if the game ends."""
    for _ in range(turns):
        payload = session.payload()
        if payload["state"]["finished"] or payload["decision"] is None:
            return
        session.act(payload["decision"]["legal_mask"].index(True))


@given(cfg=rules_configs(), seed=st.integers(0, 5_000))
@SETTINGS
def test_a_hint_is_always_a_move_the_human_could_make(cfg, seed):
    rules = Rules.from_dict(cfg)
    session = Session(rules, human_seat=1, seed=seed)
    session.hint_particles = 48

    asked = 0
    for _ in range(30):
        payload = session.payload()
        if payload["state"]["finished"] or payload["decision"] is None:
            break
        reply = session.hint()
        assert reply["hint"] is not None
        assert reply["hint"]["seat"] == 1
        assert payload["decision"]["legal_mask"][reply["hint"]["action"]]
        assert reply["ms"] >= 0.0
        asked += 1
        session.act(payload["decision"]["legal_mask"].index(True))
    assert asked > 0


def test_asking_when_there_is_nothing_to_decide_answers_rather_than_raising(real_rules):
    """The client polls this, so an exception here would surface as a dead panel."""
    session = Session(real_rules, human_seat=0, seed=3)
    play_out(session)

    reply = session.hint()
    assert reply["hint"] is None
    assert reply["reason"]


def test_the_belief_is_fed_by_watching_not_by_being_asked(real_rules):
    """No hint is ever requested here, and the belief must still know about passes.

    A pass is only visible as a difference between two consecutive views: a card
    that arrived on the table and stayed there is one every other seat declined or
    was ineligible for. So `pass_events` can only be non-empty if the advisor saw
    the quiet positions too, which makes this the direct test of that wiring - and
    it is the signal humans do not track, so it is most of the belief's edge.

    Note it is inferred from the table, not read off the transcript's
    `claim_passed` event. That event only fires when a seat that *could* claim
    declined, which greedy bots never do; the inference also counts ineligibility,
    which is why it sees far more.
    """
    session = Session(real_rules, human_seat=0, seed=11)
    advance(session, 12)

    belief = session.advisor.agent.belief
    assert belief is not None, "the advisor was never fed a state"
    assert belief.pass_events, (
        "the advisor saw no unclaimed discards, so it was not observing every state"
    )
    # And it is current rather than merely initialised: what it counts as seen is
    # exactly the public record of the position now on screen.
    state = session.engine.public_state(0)
    expected = [
        state.hand[s] + state.table[s] + state.scored[s] for s in range(len(state.hand))
    ]
    assert list(belief.seen) == expected


def test_asking_twice_for_the_same_position_does_not_shift_the_belief(real_rules):
    """The Ask button can be pressed repeatedly, and evidence must not compound.

    Folding the same view in twice would double-count it — the advice would drift
    with the number of clicks, which is invisible and would read as noise.
    """
    session = Session(real_rules, human_seat=0, seed=11)
    advance(session, 12)
    belief = session.advisor.agent.belief

    session.hint()
    seen, passes = list(belief.seen), len(belief.pass_events)
    for _ in range(3):
        session.hint()

    assert list(belief.seen) == seen
    assert len(belief.pass_events) == passes


def test_changing_the_sample_count_keeps_the_accumulated_belief(real_rules):
    """Why hint() retunes the agent instead of building a new one.

    Rebuilding would be the obvious way to change particle counts and would throw
    away every pass observed so far, mid-game, for a UI setting.
    """
    session = Session(real_rules, human_seat=0, seed=11)
    advance(session, 12)
    before = session.advisor.agent.belief
    passes = len(before.pass_events)
    assert passes > 0

    session.hint_particles = 256
    session.hint()

    assert session.advisor.agent.belief is before
    assert len(before.pass_events) == passes
    assert session.advisor.agent.particles == 256
