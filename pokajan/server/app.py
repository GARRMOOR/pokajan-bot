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
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

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
        "text": "Deck composition: 100 cards spread as evenly as possible across "
                "all character/colour slots.",
        "check": "Does the game show you how many of each card remain? If so we can "
                 "count exactly instead of inferring.",
        "impact": "high",
    },
    {
        "id": "discard_after_in_turn_call",
        "text": "Calling on your own turn still ends with a discard.",
        "check": "Call a Pokajan on your own turn. Are you then asked to discard?",
        "impact": "medium",
    },
    {
        "id": "deal_size",
        "text": "Opening hand is 7 cards, same as the hand limit.",
        "check": "Count your cards at the start of a round.",
        "impact": "low",
    },
    {
        "id": "split_rounding",
        "text": "When a self-drawn payout does not divide by three, the remainder "
                "goes to the earliest payers.",
        "check": "Self-draw a hand paying 120 or 480. Do all three pay equally?",
        "impact": "low",
    },
    {
        "id": "tiebreak_from",
        "text": "On an exact payout tie, priority runs from the discarder.",
        "check": "Rare. Only matters if two players claim the same card for the "
                 "same amount.",
        "impact": "low",
    },
]


class Session:
    """One game: a human seat, bot seats, and the history needed to replay it."""

    def __init__(self, rules: Rules, human_seat: int = 0, seed: int | None = None) -> None:
        self.rules = rules
        self.human_seat = human_seat
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
        self.history.append(
            {
                "state": asdict(self.engine.public_state(self.human_seat)),
                "events": events,
            }
        )

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
