r"""Replay a logged round through the accumulator and report what it worked out.

    .\.venv\Scripts\python scripts\replay.py                 # every log in data/games/
    .\.venv\Scripts\python scripts\replay.py data\games\X.jsonl

This is the counterpart to `check_vision.py`. That one scores the reader on single frames;
this one scores the thing above it -- whether a stream of frames adds up to a round. The two
failures are quite different: a frame reader is wrong about a card, while an accumulator is
wrong about what has happened, and only the second one is invisible in the log it wrote.

Nothing here is a test, for the same reason as `check_vision.py`: the logs are records of
specific online matches against real people and are gitignored, so the automated suite exercises
the machinery on synthetic readings and the real thing gets measured here, on the machine that
holds it. The check that matters most is the last line of each round: coins must total
4000 plus whatever the bankruptcy floor minted, and every payout must be an amount some legal
hand shape actually pays. Both come from evidence the accumulator never gets to choose.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pokajan.core.rules import load_default
from pokajan.vision.accumulate import RoundBelief
from pokajan.vision.journal import GAMES_DIR, Journal, Record, RoundTracker


def _advice(belief: RoundBelief) -> str:
    """What the advisor would say about the round's last read position.

    Printed here because this is the first place the whole chain is visible at once -- frames
    to belief to `PublicState` to a recommendation -- and because a discard that looks wrong to
    the player is the cheapest signal available that something upstream is off. The reasoning
    is often counter-intuitive on purpose (see `agents/advisor.py`), so it comes with the two
    numbers behind it rather than as a bare instruction.

    The position is the *end* of the round, which is usually not a real decision point: it is a
    sanity check that the seam carries, not a claim about what should have been played.
    """
    from pokajan.vision.state import RoundAdvisor

    view, advice = RoundAdvisor(seed=1, particles=16).advise(belief)
    if advice is None:
        return "no advice: " + ("; ".join(view.reasons) or "no card in hand this round deals")
    caveats = []
    if view.hand_unread:
        caveats.append(f"{view.hand_unread} of your cards unread")
    if view.short_by:
        caveats.append(f"{view.short_by} table cards unseen")
    short = (", " + ", ".join(caveats)) if caveats else ""
    return f"advice at the last read frame: {advice.action_label} ({advice.confidence:.0%}{short})"


def replay(path: Path) -> bool:
    """Replay one journal. True if every round in it stayed tracked."""
    rules = load_default()
    # Rounds are re-derived rather than taken from `record.round_index`, which was decided live
    # by whatever `RoundTracker` looked like on the day. The journal is a record of frames, not
    # of conclusions, so every conclusion in it should be reachable again -- and this one has
    # already been wrong once: a deck counter misread as 1 mid-payout split a round in two,
    # which restarted the coin ledger at the opening 1000 and cost the next 56 frames.
    rounds: dict[int, list[Record]] = defaultdict(list)
    tracker = RoundTracker()
    for record in Journal(path).records():
        rounds[tracker.observe(record.reading())].append(record)

    print(f"=== {path.name}: {sum(len(r) for r in rounds.values())} records, "
          f"{len(rounds)} round(s)")
    intact = True
    for index in sorted(rounds):
        records = rounds[index]
        belief = RoundBelief(rules)
        advising = 0
        for record in records:
            payout = belief.observe(record.reading())
            if payout is not None:
                print(f"  {payout}")
            # Counted per frame, not read off the end. The last frame of a round is usually
            # mid-animation and says almost nothing about whether the round was advisable while
            # it was being played -- and judging by it hid a real problem once already, when a
            # rule that kept the last *complete* hand forever looked like it was tracking while
            # advising on a hand six frames stale at the median and sixty-one at worst.
            advising += belief.tracking.ready

        bonus = (belief.bonus if belief.bonus
                 else f"{belief.bonus_by_vote} (agreed across the round, no frame named it)"
                 if belief.bonus_by_vote else "unread")
        print(f"  round {index}: {len(records)} frames, roster "
              f"{'/'.join(belief.roster) if belief.roster else 'unread'}, bonus "
              f"{bonus}, deck {belief.deck_remaining}")
        print(f"    coins {belief.ledger.latest}  total {belief.ledger.total}  "
              f"minted {belief.ledger.minted}  "
              f"{'balances' if belief.ledger.balances else 'DOES NOT BALANCE'}")
        def span(pair: tuple[int, int] | None) -> str:
            if pair is None:
                return "unknown"
            low, high = pair
            return str(low) if low == high else f"{low}-{high}"

        print(f"    {len(belief.payouts)} payout(s), {span(belief.scored_span)} cards scored "
              f"away, {span(belief.table_span)} on the table by conservation "
              f"({belief.discards.cards} of them named by the discard fields)")

        tracking = belief.tracking
        print(f"    advice available on {advising} of {len(records)} frames "
              f"({100 * advising / max(1, len(records)):.0f}%)")
        print(f"    at the last frame: {tracking}")
        for reason in tracking.reasons:
            print(f"      - {reason}")
        for reason in tracking.table_reasons:
            print(f"      table: {reason}")
        if tracking.ready:
            print(f"      {_advice(belief)}")
        intact &= tracking.ready
        print()
    return intact


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(GAMES_DIR.glob("*.jsonl"))
    if not paths:
        print(f"no logs in {GAMES_DIR} -- run scripts\\capture.py while playing a round")
        return 2
    return 0 if all([replay(path) for path in paths]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
