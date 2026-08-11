"""Advice reaching the overlay with the screen reader as the producer.

Exercised against a **real** server on a real socket rather than a test client, because there
is no `httpx` in this environment and because the thing most likely to be wrong is the part a
test client skips: a uvicorn loop on one thread and a capture loop publishing from another.

The rule these all circle is that the overlay is an accessory. A closed page, a server that
never came up, a port already taken -- none of them may stop the round being watched, because
the log is the irreplaceable artifact and the panel is not.
"""

from __future__ import annotations

import asyncio
import socket

import pytest

pytestmark = pytest.mark.invariant

websockets = pytest.importorskip("websockets")
pytest.importorskip("uvicorn")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def server():
    from pokajan.server.live import OverlayServer

    running = OverlayServer(port=free_port())
    if not running.start():
        pytest.skip(f"the overlay server did not start: {running.error}")
    try:
        yield running
    finally:
        running.stop()


def receive(port: int, publish, *, count: int = 1) -> list[dict]:
    """Attach an overlay, run `publish`, and collect what arrives."""
    import json

    async def run() -> list[dict]:
        url = f"ws://127.0.0.1:{port}/overlay/ws"
        async with websockets.connect(url) as page:
            first = json.loads(await asyncio.wait_for(page.recv(), timeout=5))
            await asyncio.sleep(0.05)          # let the server register the observer
            publish()
            got = [first]
            for _ in range(count):
                got.append(json.loads(await asyncio.wait_for(page.recv(), timeout=5)))
            return got

    return asyncio.run(run())


def test_an_overlay_that_attaches_first_hears_that_nothing_is_running(server):
    """The page must say something the moment it opens. `web/overlay.js` renders the `reason`
    when there is no hint, and a blank panel is indistinguishable from a crashed one."""
    got = receive(server.port, lambda: server.publish(
        {"type": "hint", "hint": None, "reason": "waiting for a round"}))

    assert got[0]["type"] == "hint" and got[0]["hint"] is None
    assert got[1]["reason"] == "waiting for a round"


def test_advice_published_from_another_thread_reaches_the_page(server):
    """The whole point. uvicorn owns a loop on its own thread and the capture loop is
    synchronous and owns itself, so this crosses between them on every frame that advises."""
    message = {"type": "hint", "hint": {"action_label": "discard Gawr Gura (blue)",
                                        "confidence": 0.62, "alternatives": [],
                                        "reasoning": "because"},
               "ms": 9.0, "particles": 48, "turn_index": 3, "decision": "DISCARD"}

    got = receive(server.port, lambda: server.publish(message))

    assert got[-1]["hint"]["action_label"] == "discard Gawr Gura (blue)"
    assert got[-1]["particles"] == 48


def test_publishing_with_nothing_attached_is_not_an_error(server):
    """The ordinary state: the watcher runs whether or not anyone opened the window."""
    assert not server.watching
    server.publish({"type": "hint", "hint": None, "reason": "nobody is looking"})


def test_a_server_that_could_not_start_reports_it_instead_of_raising():
    """A port already in use is the likely case -- two watchers, or a browser session already
    serving. The capture loop must carry on regardless, so this returns False rather than
    throwing into it."""
    from pokajan.server.live import OverlayServer

    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen(1)
    port = taken.getsockname()[1]
    try:
        second = OverlayServer(port=port)
        assert second.start(timeout=2.0) is False
        # Named, not merely reported. uvicorn answers a bind failure with `sys.exit(1)`, a
        # BaseException that goes straight through `except Exception` -- so the first version
        # let the thread die noisily and blamed a timeout for a port clash.
        assert "already in use" in second.error
    finally:
        taken.close()


def test_the_message_is_the_shape_the_renderer_already_draws():
    """`web/overlay.js` is shared with the browser panel, so a second message shape would mean a
    second renderer to keep in step. These are the fields it actually reads."""
    from pokajan.server.live import hint_message
    from pokajan.server.protocol import Recommendation
    from pokajan.vision.state import PublicView

    rec = Recommendation(seat=0, action=4, action_label="discard AzKi (pink)",
                         confidence=0.71, risk_alpha=0.0,
                         alternatives=[{"label": "x", "behind": 12}], reasoning="why")
    message = hint_message(rec, PublicView(None), particles=48, ms=9.4)

    assert message["type"] == "hint"
    for field in ("action_label", "confidence", "alternatives", "reasoning"):
        assert field in message["hint"]
    for field in ("ms", "particles", "turn_index", "decision"):
        assert field in message


def test_nothing_to_advise_on_is_sent_with_its_reason_rather_than_dropped():
    """Silence looks exactly like a dead reader. `overlay.js` capitalises and renders `reason`,
    so a round that lost track explains itself on the panel instead of freezing on stale advice.
    """
    from pokajan.server.live import hint_message
    from pokajan.vision.state import PublicView

    message = hint_message(None, PublicView(None, reasons=("the roster has not been read",)),
                           particles=48, ms=0.0)

    assert message["hint"] is None
    assert "roster" in message["reason"]
