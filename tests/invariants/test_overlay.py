"""The overlay's two halves that can be tested without a screen.

The window itself — frameless, transparent, click-through, never focused — is Win32
behaviour that only a running desktop can confirm, and
`python -m pokajan.server.overlay --check` reports it. What is testable here is
everything that would make a *correct* window show the wrong thing: whether the
advice fan-out reaches observers, whether the two surfaces can disagree, and whether
the renderer is genuinely unable to act.

That last one is the important one. The real game is played online against real
people, so an advisory panel that could send input would be cheating with extra
steps. It is enforced by the shape of the socket rather than by remembering not to.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from pokajan.server.app import AdviceHub, Session
from pokajan.server.overlay import (
    DEFAULTS,
    MIN_HEIGHT,
    MIN_WIDTH,
    PlacementApi,
    css_size,
    load_position,
    physical_size,
)

pytestmark = pytest.mark.invariant


class FakeSocket:
    """Records what it was sent, and can be told to fail like a closed socket."""

    def __init__(self, *, broken: bool = False) -> None:
        self.sent: list[dict] = []
        self.broken = broken

    async def send_text(self, text: str) -> None:
        if self.broken:
            raise RuntimeError("socket closed")
        self.sent.append(json.loads(text))


# --------------------------------------------------------------------- hub ---

def test_nothing_is_computed_when_nothing_is_watching():
    """`attached` is what keeps an unused overlay free.

    The push path is guarded by it, so a false positive here would mean paying for a
    hint on every state change whether or not anyone can see it.
    """
    hub = AdviceHub()
    assert not hub.attached()
    asyncio.run(hub.broadcast({"type": "hint"}))   # must not raise with no observers

    hub.observers.add(FakeSocket())
    assert hub.attached()


def test_advice_reaches_every_observer():
    hub = AdviceHub()
    first, second = FakeSocket(), FakeSocket()
    hub.observers |= {first, second}

    asyncio.run(hub.broadcast({"type": "hint", "hint": None}))

    assert first.sent == second.sent == [{"type": "hint", "hint": None}]


def test_a_dead_observer_is_dropped_and_does_not_take_the_game_with_it():
    """The overlay is an accessory; the round is the thing that matters.

    A closed socket raising through `broadcast` would surface inside the browser's
    own message loop and end the game the human was playing.
    """
    hub = AdviceHub()
    alive, dead = FakeSocket(), FakeSocket(broken=True)
    hub.observers |= {alive, dead}

    asyncio.run(hub.broadcast({"type": "hint", "hint": None}))

    assert hub.observers == {alive}
    assert len(alive.sent) == 1


# ------------------------------------------------------------------ advice ---

def test_the_two_surfaces_cannot_disagree_about_a_position(real_rules):
    """One answer per position, because the advisor genuinely resamples.

    Six draws on one opening position produced three different recommended cards, so
    without this the browser panel and the overlay would routinely name different
    discards for the same hand — which would destroy the trust the reasoning text
    exists to build.
    """
    session = Session(real_rules, human_seat=0, seed=5)

    for_browser = session.hint()
    for_overlay = session.hint()

    assert for_browser == for_overlay
    assert for_browser["hint"] is not None


def test_changing_the_sample_count_re_asks_rather_than_serving_the_old_answer(real_rules):
    """The cache is keyed on the setting too, or the selector would look broken."""
    session = Session(real_rules, human_seat=0, seed=5)
    cheap = session.hint()

    session.hint_particles = 1024
    dear = session.hint()

    assert dear["particles"] == 1024
    assert cheap["particles"] == 48
    # Not necessarily a different card — just genuinely recomputed at the new budget.
    assert dear["ms"] > 0.0


# ---------------------------------------------------------------- geometry ---
#
# pywebview's geometry API is not self-consistent: the constructor's width does not
# round-trip with what the window reports, while resize() does, and the two differ by
# devicePixelRatio. The first version stored physical pixels and passed them back as
# CSS ones, so the window grew by the display's scale factor on every launch. These
# guard the conversion; --check reports the live numbers.

@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5, 2.0, 3.0])
def test_size_survives_a_save_and_restore_at_any_display_scale(scale):
    """Otherwise the overlay resizes itself a little every time it starts."""
    spot = dict(DEFAULTS)

    physical = physical_size(spot, scale)
    restored = css_size(*physical, scale)

    assert restored == (spot["css_width"], spot["css_height"])


@pytest.mark.parametrize("scale", [1.0, 2.0])
def test_the_window_cannot_be_saved_smaller_than_its_own_grip(scale):
    """A window shrunk to nothing takes the resize grip with it.

    There is no frame and no menu, so the only way back would be deleting the file by
    hand — which is a bad thing to learn about a tool at the moment you need it.
    """
    tiny = {"css_width": 1, "css_height": 1}
    assert physical_size(tiny, scale) == (MIN_WIDTH, MIN_HEIGHT)


def test_a_corrupt_position_file_is_ignored_rather_than_fatal(tmp_path, monkeypatch):
    """A bad overlay position should put the panel in the wrong corner, not stop it."""
    import pokajan.server.overlay as overlay

    bad = tmp_path / "overlay.json"
    monkeypatch.setattr(overlay, "POSITION_FILE", bad)

    bad.write_text("{ not json at all")
    assert load_position() == DEFAULTS

    # Partial and wrongly-typed entries fall back per key rather than wholesale, so a
    # hand-edited file keeps whatever was still valid.
    bad.write_text('{"x": 900, "css_width": "wide"}')
    assert load_position() == {**DEFAULTS, "x": 900}


def test_the_resize_api_holds_the_window_privately():
    """The underscore is load-bearing, and nothing about it looks load-bearing.

    pywebview walks this object to decide what to expose and recurses into any public
    non-callable it finds. A public window reference sends it into the Window, which
    calls evaluate_js before startup and fails the whole js_api injection — leaving
    the grip silently inert.
    """
    api = PlacementApi()

    exposed = [name for name in dir(api) if not name.startswith("_")]
    assert exposed == ["resize"]


def test_the_resize_api_clamps_without_a_window_attached():
    api = PlacementApi()
    assert api.resize(10, 10) == {"width": MIN_WIDTH, "height": MIN_HEIGHT}
    assert api.resize(640, 300) == {"width": 640, "height": 300}


# -------------------------------------------------------------------- cache ---

def test_a_new_game_does_not_serve_the_previous_ones_advice(real_rules):
    """A cache keyed on position alone would collide across games.

    Seeds 7 and 8 are specific: both open on a DISCARD at turn 0, so the cache key
    matches exactly and only clearing it on `new_game` prevents a hit. Picked after a
    first attempt used seeds whose opening decisions differed, which made the test
    pass while never exercising the collision it claims to cover.

    The failure it guards against looks like the bot ignoring the cards in front of
    it, which is much harder to recognise as a caching bug.
    """
    session = Session(real_rules, human_seat=0, seed=7)
    first = session.hint()

    session.new_game(seed=8)
    second = session.hint()

    assert first["turn_index"] == second["turn_index"] == 0
    assert first["decision"] == second["decision"] == "DISCARD", "keys must collide"
    assert second is not first, "served the previous game's cached answer"
    # And it is advice about the hand actually being held now.
    hand = session.engine.public_state(0).hand
    assert hand[second["hint"]["action"]] > 0
