"""The playable web table.

This exists to answer one question before any training compute is spent: **does
our simulation actually match the real game?** Everything here is bent toward
making that comparison easy — a readable transcript, every payout shown with its
arithmetic, and the rules we are still guessing about listed on screen rather than
buried in a YAML comment.

    python -m pokajan.server.app

The human takes one seat and simple bots take the rest. Bots are deliberately
crude at this stage: the point is to generate lots of Pokajans so payouts can be
eyeballed, not to play well.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..agents.advisor import Advisor
from ..agents.base import GreedyCallerAgent
from ..core.engine import Engine
from ..core.rules import Rules, load_default
from ..server.protocol import ActionResponse

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

# Assumptions the GUI surfaces so they get checked while playing, rather than
# being discovered wrong after a training run. Kept next to the code that shows
# them; the authoritative TODO(M2) markers live in rules/pokajan_v1.yaml.
OPEN_QUESTIONS = [
    {
        "id": "composition",
        "text": "Which 100 of the 153 possible cards are in the deck — assumed "
                "spread as evenly as the 3-per-slot cap allows.",
        "check": "Not checkable by looking. The game's remaining-cards list counts "
                 "everything you have not seen out of the full 9-per-holomem pool, "
                 "so for a 17-holomem lineup it lists 153 cards when only 100 are "
                 "real. Roughly a third of that list does not exist.",
        "impact": "high",
    },
]

# Everything below has been confirmed by playing a real round. Kept on screen so
# the game can be spot-checked against them, rather than quietly assumed correct.
SETTLED = [
    ("Deck is always exactly 100 cards, whatever the roster size.", ""),
    ("Payout table: triple 120/840, 3-group 180/480, 4-group 300/840, "
     "5-group 480/1800 (multi/single colour).", ""),
    ("Bonus holomem adds +90 per copy the hand actually scores.", ""),
    ("Calling on your own turn still ends with a discard.", ""),
    ("A call refills you to the size you had before it — eight in turn, seven on "
     "a claim.", "Refilling flat to seven would cost you a card permanently."),
    ("On a payout tie, priority runs in turn order from the player after the "
     "discarder.", ""),
    ("No payout is indivisible by three, so a split never leaves an odd coin.", ""),
]


# Particle counts offered in the hint panel. 48 is the measured default; 1024 is
# the one configuration that beat it (+50 coins/game, see the README's M4 table) and
# is here so the cost of that edge is visible rather than asserted. The real game
# allows roughly ten seconds a turn, so the panel showing wall time is what decides
# whether the overlay can afford the expensive setting.
#
# Zero is deliberately not offered. With no particles the danger term vanishes, the
# ranking becomes purely offensive and therefore perfectly stable to resampling, so
# the reported confidence would be high for the wrong reason -- the one thing this
# panel must never do.
HINT_PARTICLES = [48, 256, 1024]
DEFAULT_HINT_PARTICLES = 48


class Session:
    """One game: a human seat, bot seats, and the history needed to replay it."""

    def __init__(self, rules: Rules, human_seat: int = 0, seed: int | None = None) -> None:
        self.rules = rules
        self.human_seat = human_seat
        self.hint_particles = DEFAULT_HINT_PARTICLES
        # One advisor for the whole session. It must outlive individual decisions
        # because its belief is accumulated from consecutive views -- see
        # Advisor.observe. Rebuilding it per request would silently discard the
        # pass-inference that is most of its edge.
        self.advisor = Advisor(rules, seed=0, particles=self.hint_particles)
        self.new_game(seed)

    def new_game(self, seed: int | None = None) -> None:
        self.engine = Engine.new_game(self.rules, seed=seed, record_events=True)
        self.bots = {
            seat: GreedyCallerAgent(self.rules, seed=(seed or 0) * 10 + seat)
            for seat in range(self.rules.play.players)
            if seat != self.human_seat
        }
        self.history: list[dict] = []
        self._events_sent = 0
        self._run_bots()
        self._snapshot()

    # ------------------------------------------------------------- driving --
    def _run_bots(self) -> None:
        """Let the bots act until the human is needed, or the game ends.

        Bot answers inside a claim window are submitted as they come, which is
        safe: the engine records claim responses without mutating anything until
        every eligible seat has answered, so the human's request is still built
        from the same untouched state.
        """
        guard = 0
        while not self.engine.finished:
            pending = self.engine.pending_decisions()
            if not pending:
                break
            bot_requests = [r for r in pending if r.seat != self.human_seat]
            if not bot_requests:
                break
            self.engine.submit(
                [self.bots[r.seat].act(r) for r in bot_requests]
            )
            guard += 1
            if guard > 10_000:
                raise RuntimeError("bots are not making progress")

    def act(self, action: int) -> None:
        pending = self.engine.pending_decisions()
        mine = [r for r in pending if r.seat == self.human_seat]
        if not mine:
            raise ValueError("it is not your turn")
        self.engine.submit(
            ActionResponse(
                game_id=self.engine.state.game_id, seat=self.human_seat, action=action
            )
        )
        self._run_bots()
        self._snapshot()

    # ------------------------------------------------------------- viewing --
    def _snapshot(self) -> None:
        events = self.engine.state.events[self._events_sent:]
        self._events_sent = len(self.engine.state.events)
        view = self.engine.public_state(self.human_seat)
        # Fed on every snapshot, not only when a hint is asked for. Whether anyone
        # claimed a discard is only readable as a difference between consecutive
        # views, so a belief that skipped the quiet positions would be inferring
        # from gaps. Costs nothing -- observe() does no sampling.
        self.advisor.observe(view)
        self.history.append({"state": asdict(view), "events": events})

    # -------------------------------------------------------------- hinting --
    def hint(self) -> dict:
        """What the bot would do here, with the time it took to decide.

        On demand rather than bundled into every update, for two reasons: the
        expensive part should not sit on the critical path of ordinary play, and a
        hint that appears unasked turns the validation table into a spectator sport
        just when odd branches most need exercising by hand.

        The wall time is part of the answer, not diagnostics. The real game allows
        about ten seconds a turn, and whether the overlay can afford 1024 particles
        is a question only a measurement answers.
        """
        mine = next(
            (r for r in self.engine.pending_decisions() if r.seat == self.human_seat),
            None,
        )
        if mine is None:
            return {"type": "hint", "hint": None, "reason": "nothing to decide"}

        self.advisor.agent.particles = self.hint_particles
        started = time.perf_counter()
        rec = self.advisor.recommend(mine)
        elapsed = (time.perf_counter() - started) * 1000.0
        return {
            "type": "hint",
            "hint": asdict(rec),
            "ms": round(elapsed, 1),
            "particles": self.hint_particles,
            "turn_index": mine.state.turn_index,
            "decision": mine.decision,
        }

    def payload(self) -> dict:
        pending = self.engine.pending_decisions()
        mine = next((r for r in pending if r.seat == self.human_seat), None)
        waiting_on = sorted(r.seat for r in pending)
        return {
            "type": "update",
            "setup": asdict(self.engine.setup()),
            "human_seat": self.human_seat,
            "state": asdict(self.engine.public_state(self.human_seat)),
            "decision": asdict(mine) if mine else None,
            "waiting_on": waiting_on,
            "history": self.history,
            "open_questions": OPEN_QUESTIONS,
            "settled": [{"text": t, "note": n} for t, n in SETTLED],
            "hint_particles": HINT_PARTICLES,
            "hint_particles_current": self.hint_particles,
            # Shown on screen so the numbers can be read straight off the config
            # and compared against the real game without opening the YAML.
            "payout_table": self.rules.raw["payouts"]["table"],
            "assumptions": {
                "discard_after_in_turn_call": self.rules.play.discard_after_in_turn_call,
                "deal_size": self.rules.play.deal_size,
                "composition": self.rules.composition.value,
                "tiebreak_from": self.rules.tiebreak_from,
                "bonus_per_copy": self.rules.bonus_per_copy,
                "bonus_applies_to": self.rules.bonus_applies_to,
            },
        }


def create_app(rules: Rules, human_seat: int = 0) -> FastAPI:
    app = FastAPI(title="Pokajan")

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        page = WEB_DIR / "index.html"
        if not page.exists():
            return HTMLResponse("<h1>web/index.html is missing</h1>", status_code=500)
        return FileResponse(page)

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        session = Session(rules, human_seat=human_seat, seed=None)
        await socket.send_text(json.dumps(session.payload()))
        try:
            while True:
                message = json.loads(await socket.receive_text())
                kind = message.get("type")
                if kind == "action":
                    try:
                        session.act(int(message["action"]))
                    except ValueError as exc:
                        await socket.send_text(
                            json.dumps({"type": "error", "message": str(exc)})
                        )
                        continue
                elif kind == "newgame":
                    session.new_game(message.get("seed"))
                elif kind == "hint":
                    # Answered on its own rather than by resending the whole state,
                    # so asking for advice never disturbs the table -- the overlay
                    # will be doing exactly this against a game it cannot touch.
                    if "particles" in message:
                        want = int(message["particles"])
                        if want not in HINT_PARTICLES:
                            await socket.send_text(json.dumps(
                                {"type": "error", "message": f"bad particle count {want}"}
                            ))
                            continue
                        session.hint_particles = want
                    await socket.send_text(json.dumps(session.hint()))
                    continue
                else:
                    await socket.send_text(
                        json.dumps({"type": "error", "message": f"unknown message {kind!r}"})
                    )
                    continue
                await socket.send_text(json.dumps(session.payload()))
        except WebSocketDisconnect:
            return

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description="Play Pokajan in a browser")
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--seat", type=int, default=0, help="which seat you play")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    rules = Rules.load(args.rules) if args.rules else load_default()
    print(f"rules   {rules.name} ({rules.rules_hash[:12]})")
    print(f"roster  {rules.cards.n_chars} characters, groups "
          f"{[len(m) for m in rules.cards.group_members]}")
    print(f"\n  http://{args.host}:{args.port}\n")

    import uvicorn

    uvicorn.run(create_app(rules, args.seat), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
