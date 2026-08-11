"""The seam: what the screen reader accumulated, as the state the agents already speak.

The failure this file guards is quiet. A `PublicState` with a plausible-looking but wrong
`table` does not raise anything -- it produces confident advice about a card the belief thinks
is still out there when it has already been watched leaving. So most of what is pinned here is
the *direction* of every approximation: short, never long.

Readings are synthetic. Real rounds live in `data/games/`, are gitignored, and are measured by
`scripts/replay.py` on the machine that holds them.
"""

from __future__ import annotations

import pytest

from pokajan.core.rules import load_default
from pokajan.vision.accumulate import RoundBelief
from pokajan.vision.reader import Card, FrameReading
from pokajan.vision.state import UNREAD_FIELDS, public_view, round_rules

pytestmark = pytest.mark.invariant

SEATS = ("bottom", "left", "top", "right")
ROSTER = ("gen1", "gen4", "holox", "id1")


@pytest.fixture(scope="module")
def rules():
    return load_default()


def reading(**kwargs) -> FrameReading:
    base = dict(
        at=0.0,
        screen="table",
        deck_remaining=60,
        coins={seat: 1000 for seat in SEATS},
        hand=tuple(Card("tokoyami_towa", "blue") for _ in range(7)),
        roster=ROSTER,
        bonus="tokoyami_towa",
    )
    base.update(kwargs)
    return FrameReading(**base)


def settled(rules, **kwargs) -> RoundBelief:
    belief = RoundBelief(rules)
    belief.observe(reading(**kwargs))
    return belief


# ------------------------------------------------------- the round's cards ----
def test_the_card_space_is_this_rounds_roster_and_not_a_catalogue(rules):
    """A slot index only means anything relative to the roster in play, and the roster is
    redrawn every round. `rules/pokajan_v1.yaml` describes *one* round for exactly that reason,
    so a round read off the screen has to build its own."""
    space = round_rules(ROSTER).cards

    assert space.group_ids == ROSTER
    assert space.n_slots == len(space.character_ids) * space.n_colors
    assert "tokoyami_towa" in space.character_ids           # Gen4
    assert "tokino_sora" not in space.character_ids         # Gen0, not dealt this round


def test_every_number_but_the_roster_still_comes_from_the_rules_file(rules):
    """The substitution must not fork the payout table. One place for gameplay numbers is the
    rule this project is built on, and a second copy here would be the obvious way to break
    it -- a corrected payout would then be right in the engine and wrong in the overlay."""
    built = round_rules(ROSTER)

    from pokajan.core.rules import HandKind

    assert built.deck_size == rules.deck_size
    assert built.play.hand_limit == rules.play.hand_limit
    assert built.play.initial_coins == rules.play.initial_coins
    # Every cell that does not depend on the roster, not just one: a fork would most likely
    # show up in whichever row nobody thought to check.
    for kind, size in ((HandKind.TRIPLE, None), (HandKind.GROUP, 3),
                       (HandKind.GROUP, 4), (HandKind.GROUP, 5)):
        for mono in (False, True):
            for bonus in (0, 1, 3):
                assert built.payout(kind, group_size=size, monochrome=mono,
                                    bonus_copies=bonus) == \
                    rules.payout(kind, group_size=size, monochrome=mono, bonus_copies=bonus)


def test_a_roster_the_book_does_not_know_is_refused_rather_than_dropped():
    with pytest.raises(KeyError, match="roster names groups"):
        round_rules(("gen4", "not_a_real_group"))


# ------------------------------------------------------ short, never long ----
def test_the_table_holds_only_cards_actually_seen(rules):
    """`envs/belief.py` reads `hand + table + scored` as "cards accounted for" and treats the
    rest as unseen, so an over-count makes the agent confidently wrong about a card it has
    already watched leave. Under-counting only makes it less certain. Two seats' discard
    staircases are unread and four turns in five pass unobserved, so this is short constantly
    and must stay short rather than being topped up to what conservation allows."""
    thrown = (Card("tokoyami_towa", "pink"), Card("amane_kanata", "blue"))
    belief = settled(rules, discards={"bottom": thrown})

    view = public_view(belief)

    assert view.state is not None
    assert sum(view.state.table) == 2, "exactly the cards read, not the conservation bound"


def test_the_shortfall_is_reported_rather_than_hidden(rules):
    """A caller showing advice needs to know how much of the table went unseen, or it cannot
    say how much to trust it. Conservation knows the count even where no reader named them."""
    belief = settled(rules, discards={"bottom": (Card("tokoyami_towa", "pink"),)})

    view = public_view(belief)

    assert view.state is not None
    low, _ = belief.table_span
    assert view.short_by == max(0, low - 1)
    assert view.short_by > 0, "the fixture must actually have unseen cards, or this proves nothing"


def test_a_card_the_round_cannot_deal_is_dropped_not_counted(rules):
    """A misread naming a holomem outside the roster would otherwise land in some slot or
    raise. Dropping it keeps the vector a lower bound, which is the one property the belief
    depends on."""
    intruder = (Card("tokino_sora", "blue"), Card("tokoyami_towa", "pink"))
    belief = settled(rules, discards={"bottom": intruder})

    view = public_view(belief)

    assert view.state is not None
    assert sum(view.state.table) == 1, "tokino_sora is Gen0 and this round deals Gen1/4/HoloX/ID1"


# ------------------------------------------------------------- the refusal ----
def test_a_round_that_lost_track_yields_no_state_at_all(rules):
    """Advice from a broken accumulation is worse than silence, because it looks the same as
    advice from a sound one."""
    belief = RoundBelief(rules)
    belief.observe(reading(roster=None, bonus=None))

    view = public_view(belief)

    assert view.state is None
    assert view.reasons


def test_the_fields_with_no_reader_are_named_even_when_a_state_is_produced(rules):
    """`phase`, `last_discard` and `turn_index` are filled with placeholders because the
    dataclass requires them, and a placeholder that is not announced is a lie the overlay would
    render as fact. Naming them is what lets it show its own gaps."""
    view = public_view(settled(rules))

    assert view.state is not None
    assert set(view.unknown) == set(UNREAD_FIELDS)
    assert "last_discard" in view.unknown
    assert "current_seat" not in view.unknown, "the turn indicator has a reader now"


def test_the_turn_carries_through_with_the_time_it_was_last_seen(rules):
    """The bars are covered by every payout panel, which is exactly the moment the watcher
    samples hardest, so the answer is routinely a turn or two old. Carrying it forward is worth
    far more to an overlay than refusing — but only if how stale it is comes with it."""
    belief = settled(rules, at=12.0, current_seat="left")
    belief.observe(reading(at=13.0, current_seat=None))      # a payout covers the bars

    view = public_view(belief)

    assert view.state is not None
    assert view.state.current_seat == SEATS.index("left")
    assert view.turn_read_at == 12.0


def test_a_round_that_never_saw_a_lit_bar_says_so_rather_than_implying_a_turn(rules):
    """`PublicState.current_seat` is not optional, so it has to hold something. The viewer is
    the least misleading filler, and `turn_read_at` of None is what marks it as filler rather
    than a reading."""
    view = public_view(settled(rules, current_seat=None))

    assert view.state is not None
    assert view.turn_read_at is None


def test_the_viewer_is_the_seat_the_capture_is_taken_from(rules):
    """The screenshot is of this machine, so the hand along the bottom is the player's own."""
    view = public_view(settled(rules))

    assert view.state is not None
    assert view.state.viewer == SEATS.index("bottom")
    assert sum(view.state.hand) == 7


# ------------------------------------------------------ the live advisor ----
def test_the_per_round_setup_happens_once_not_once_a_frame(rules):
    """Building a `Rules` for a roster re-parses the rules file and costs 19 ms against the 9 ms
    the advice itself takes, so rebuilding it per frame would make the setup twice the work --
    on the loop that was expensive to get down to a second."""
    from pokajan.vision.state import RoundAdvisor

    built: list[tuple[str, ...]] = []
    advisor = RoundAdvisor(particles=8, seed=1)
    original = advisor.rules_for

    def counting(roster):
        built.append(roster)
        return original(roster)

    advisor.rules_for = counting
    belief = settled(rules)
    for _ in range(5):
        advisor.advise(belief)

    assert len(set(built)) == 1
    assert len(advisor._rules) == 1, "one Rules for the round, not one per frame"


def test_one_advisor_per_round_so_the_belief_accumulates(rules):
    """`agents/advisor.py` is explicit that an unclaimed discard is only visible as a difference
    between two consecutive views, so a fresh `Advisor` per frame would be blind to the very
    signal its belief is built on. Different rounds must still get their own."""
    from pokajan.vision.state import RoundAdvisor

    advisor = RoundAdvisor(particles=8, seed=1)
    advisor.advise(settled(rules))
    advisor.advise(settled(rules))
    assert len(advisor._advisors) == 1

    other = ("gen0", "gen4", "holox", "id1")
    advisor.advise(settled(rules, roster=other))

    assert len(advisor._advisors) == 2, "a new roster is a new round and a new belief"


def test_a_round_with_nothing_to_say_still_returns_its_view(rules):
    """"No advice, and here is why" is the answer the overlay needs on frames it cannot read.
    Silence looks identical to a reader that has crashed."""
    from pokajan.vision.state import RoundAdvisor

    belief = RoundBelief(rules)
    belief.observe(reading(roster=None, bonus=None))

    view, advice = RoundAdvisor(particles=8, seed=1).advise(belief)

    assert advice is None
    assert view.state is None and view.reasons


# ------------------------------------------------------------ end to end ----
def test_the_advisor_accepts_what_the_reader_produces(rules):
    """The one test that proves the seam rather than describing it.

    `agents/advisor.py` promises that at M8 the screen reader becomes another source of
    `PublicState` "and nothing here changes". This is that claim, executed: a state built
    entirely from frame readings, through the real advisor, to an action.
    """
    from pokajan.agents.advisor import Advisor
    from pokajan.server.protocol import DecisionRequest

    hand = tuple(Card(name, "blue") for name in
                 ("tokoyami_towa", "amane_kanata", "himemori_luna", "tsunomaki_watame",
                  "shirakami_fubuki", "akai_haato", "aki_rosenthal"))
    belief = settled(rules, hand=hand)
    view = public_view(belief)
    assert view.state is not None

    built = round_rules(ROSTER)
    advisor = Advisor(built, seed=1, particles=8)
    mask = [view.state.hand[slot] > 0 for slot in range(built.cards.n_slots)] + [False] * 3
    request = DecisionRequest(game_id=view.state.game_id, seat=view.state.viewer,
                              decision="DISCARD", legal_mask=mask, state=view.state)

    advice = advisor.recommend(request)

    assert 0 <= advice.action < built.cards.n_slots
    assert view.state.hand[advice.action] > 0, "it must discard a card actually held"
    assert advice.action_label
