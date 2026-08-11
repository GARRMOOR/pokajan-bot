"""Serving the overlay with the screen reader as the producer.

`server/app.py` says what this is for: "when the screen reader arrives it becomes the producer
and the renderer does not change". This is that swap, and the renderer does not change --
`web/overlay.js` cannot tell whether the `hint` message it draws came from a browser game or
from a watcher pointed at the real one.

The overlay window is a separate process by necessity: pywebview owns a main loop and so does
uvicorn. So the shape is two commands, not one --

    .\\.venv\\Scripts\\python scripts\\capture.py --overlay      # watch, advise, serve
    .\\.venv\\Scripts\\python -m pokajan.server.overlay          # the window itself

-- and this module is the first of them. It runs uvicorn on a daemon thread so the capture loop
stays exactly as it was, in charge and synchronous, and `publish` hands a message across from
that thread.

**Nothing here can act on the game.** The socket is one-way by construction: `/overlay/ws`
discards whatever the page sends, and this class has no path from a rendered panel back to
anything. That is the same rule the rest of M8 is built on -- the game is played online against
real people, and an advisor that could click is a second player.

**A dead overlay must never disturb the round.** Publishing to a closed socket, a server that
failed to start, or a port already in use all end the same way: the watcher keeps reading and
the log keeps being written. The overlay is an accessory.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import asdict
from typing import Any


class OverlayServer:
    """Serves the overlay page and pushes advice to it, from another thread.

    Started lazily and stopped on exit. `publish` is safe to call from the capture loop and
    never raises: an overlay nobody opened, a page that was closed, or a port already taken are
    all ordinary states rather than errors, and none of them is a reason to stop watching.
    """

    def __init__(self, *, port: int = 8000, host: str = "127.0.0.1") -> None:
        self.port = port
        self.host = host
        self.error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._hub: Any = None
        self._ready = threading.Event()

    # ------------------------------------------------------------- lifecycle --
    def start(self, timeout: float = 10.0) -> bool:
        """Bring the server up on its own thread. False, with `error` set, if it could not."""
        try:
            import uvicorn

            from .app import create_app
            from ..core.rules import load_default
        except ImportError as exc:                       # pragma: no cover - env-specific
            self.error = f"{exc} -- the overlay needs fastapi and uvicorn"
            return False

        # A `Rules` is required to build the app, and the overlay routes never touch it: the
        # only producer here is the reader, which brings its own per-round rules. Passing the
        # default keeps `create_app` honest rather than making its argument optional for one
        # caller's convenience.
        app = create_app(load_default())
        self._hub = getattr(app.state, "hub", None)
        if self._hub is None:
            self.error = "the app exposes no advice hub to publish through"
            return False

        config = uvicorn.Config(app, host=self.host, port=self.port,
                                log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._ready.set()
            try:
                self._loop.run_until_complete(self._server.serve())
            except SystemExit:
                # uvicorn's own response to a port it cannot bind is `sys.exit(1)`, which is a
                # BaseException and so sails straight through `except Exception` -- leaving the
                # thread to die noisily while `start` blamed a timeout for a bind failure. The
                # port being taken is the *likely* case here (a browser session already serving,
                # or a second watcher), so it is worth naming exactly.
                self.error = (f"port {self.port} is already in use -- another overlay server, "
                              f"or the browser app, is on it")
            except Exception as exc:                     # pragma: no cover - env-specific
                self.error = str(exc)
            finally:
                self._ready.set()

        self._thread = threading.Thread(target=run, name="overlay-server", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        if self.error:
            return False
        # `started` is uvicorn's own flag and it lags the thread by a moment.
        deadline = threading.Event()
        for _ in range(int(timeout * 20)):
            if getattr(self._server, "started", False):
                return True
            deadline.wait(0.05)
        self.error = self.error or f"the server did not come up on {self.host}:{self.port}"
        return False

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    # -------------------------------------------------------------- publishing --
    @property
    def watching(self) -> bool:
        """Whether any overlay is actually attached, so advice for nobody costs nothing."""
        return bool(self._hub and self._hub.attached())

    def publish(self, message: dict) -> None:
        """Hand one message to every attached overlay. Never raises, never blocks for long."""
        if not self._hub or self._loop is None or self._loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._hub.broadcast(message), self._loop)
            future.result(timeout=1.0)
        except Exception:
            # A dead socket, a stopping loop, a slow renderer. The round is what matters.
            pass


def hint_message(advice, view, *, particles: int, ms: float) -> dict:
    """One `Recommendation` in the shape `web/overlay.js` already draws.

    Deliberately the same message the browser session sends, field for field, because the
    renderer is shared and a second shape would mean a second renderer to keep in step.

    `turn_index` is carried through as whatever the state holds, which from the reader is always
    0 -- nothing counts turns, and `state.UNREAD_FIELDS` says so. It is sent anyway because the
    browser session sends it and the message shapes must not diverge. Nothing renders it:
    `web/overlay.js` decides staleness from when a message *arrived*, which is the right measure
    for a producer that pushes rather than being polled.
    """
    if advice is None:
        return {"type": "hint", "hint": None,
                "reason": "; ".join(view.reasons) if view.reasons else "nothing to advise on"}
    return {
        "type": "hint",
        "hint": asdict(advice),
        "ms": round(ms, 1),
        "particles": particles,
        "turn_index": view.state.turn_index if view.state else 0,
        "decision": "DISCARD",
    }
