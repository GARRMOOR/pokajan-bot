r"""Watch the game, read each settled frame, write down what was seen, keep no pictures.

    .\.venv\Scripts\python scripts\capture.py                 # watch until Ctrl-C
    .\.venv\Scripts\python scripts\capture.py --once          # one frame, then stop
    .\.venv\Scripts\python scripts\capture.py --frame <path>  # read a saved capture instead
    .\.venv\Scripts\python scripts\capture.py --status        # what is on disk right now
    .\.venv\Scripts\python scripts\capture.py --purge         # delete every kept crop
    .\.venv\Scripts\python scripts\capture.py --overlay       # and feed the overlay window
    .\.venv\Scripts\python scripts\capture.py --no-advice     # log only, say nothing

With `--overlay` this also serves the window's page and pushes advice to it, so the whole thing
is two commands: this one, and `python -m pokajan.server.overlay` for the window itself. The
socket is one-way and the window is click-through, so neither can reach the game.

Prints what to discard while the round is still being played, on the frames it can read. That
costs 9 ms against the 285 ms a read takes, so it runs whenever the round is tracking -- but it
only prints when the answer *changes*, because a line repeated every second is a line nobody
reads. `--no-advice` turns it off.

**No screenshot is ever written.** A frame is grabbed into memory, read, and dropped; the
history lives in `data/games/<stamp>.jsonl` as text. That is stronger than deleting a file
afterwards -- there is no moment at which a picture of a live online match exists on disk.

The one exception is small crops of the regions whose readers are not built yet -- the newest
discard in each field -- because their content cannot be recovered from a log that only holds
what something already knows how to read. Those go to `data/pending/`, under a byte cap, from
an allowlist of regions that carry no usernames, and `--purge` removes them. `--no-crops`
turns even that off.

Read-only, and that is the entire scope of M8: nothing here sends input to the game.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from pokajan.core.rules import load_default
from pokajan.vision import layout
from pokajan.vision.accumulate import RoundBelief
from pokajan.vision.capture import (
    IDLE_HEARTBEAT,
    PAYOUT_HEARTBEAT,
    Grab,
    MonitorSource,
    PendingStore,
    TurnGate,
    crops_due,
    frames,
)
from pokajan.vision.journal import Journal, Record, RoundTracker
from pokajan.vision.reader import FrameReading, TableReader, deck_key
from pokajan.server.live import OverlayServer, hint_message
from pokajan.vision.state import RoundAdvisor

# Regions kept as crops while their readers are unbuilt. The four discard fields, because the
# newest card in each is what a continuous reader actually needs -- one per turn, at the end
# furthest from its seat -- and no amount of log holds it until that reader exists.
UNREAD_REGIONS = {f"discards_{seat}": layout.DISCARDS[seat] for seat in layout.SEAT_ORDER}

# Regions kept when the read of them *refused*, so a failure can be looked at instead of
# guessed about. Keyed by the prefix of the refusal each one explains.
#
# This is the crop rule that should have been here from the start. Keeping the same regions
# every frame filled 48 MB with pictures of reads that worked, while the thing that actually
# needed looking at -- a hand whose fifth card refused on 72 of 77 frames -- was never kept at
# all. A successful read needs no evidence; it is already in the log.
#
# Every region here must be on `capture.CROPPABLE`, which excludes the coin boxes because those
# reach over the player's name. Coin refusals are diagnosed by `digits.describe` as text
# instead -- see `--tune`.
REFUSAL_REGIONS = {
    "hand": ("hand", layout.HAND),
    "bonus": ("bonus_card", layout.BONUS_CARD),
    "roster": ("group_labels", layout.GROUP_LABELS),
    "deck_remaining": ("deck_counter", layout.DECK_COUNTER),
}


class LiveAdvice(RoundAdvisor):
    """The advisor, plus the last thing it said, so an unchanged answer stays quiet.

    Subclassed rather than wrapped because the per-round caching is the whole reason
    `RoundAdvisor` exists and there is nothing here to keep separate from it.

    `overlay` is the window, when one is being served. It is fed on **every** advised frame
    while the console line is printed only on a change: a panel showing a hint from a minute
    ago looks exactly like a panel showing the current one, so `overlay.js` fades a stale
    message -- and it can only do that if messages keep arriving.
    """

    last: str | None = None
    overlay: OverlayServer | None = None


def refusal_shape(reading: FrameReading) -> frozenset[str]:
    """Which *kinds* of refusal a frame had, with card positions kept but indices generalised.

    Used to keep one crop per distinct failure rather than one per frame. Position is retained
    because it is the diagnostic signal: card 5 refusing on 72 of 77 frames while cards 0 to 3
    read cleanly is not an art-coverage problem, and that pattern is only visible if the
    position survives.
    """
    shape = set()
    for key in reading.refusals:
        for prefix in REFUSAL_REGIONS:
            if key.startswith(prefix):
                shape.add(key.removesuffix("_colour"))
                break
    return frozenset(shape)


def summarise(record: Record, reading: FrameReading, belief: RoundBelief | None) -> str:
    parts = [f"r{record.round_index}", f"[{reading.screen}]"]
    parts.append(f"deck {reading.deck_remaining}" if reading.deck_remaining is not None
                 else "deck ?")
    total = reading.coins_total
    parts.append(f"coins {total}" if total is not None
                 else f"coins {len(reading.coins)}/4")
    if reading.roster:
        parts.append("/".join(reading.roster))
    known = sum(1 for card in reading.hand if card.known)
    if reading.hand:
        parts.append(f"hand {known}/{len(reading.hand)}")
    if reading.refusals:
        parts.append(f"({len(reading.refusals)} refused)")
    if belief is not None:
        # The one thing a watcher needs at a glance, since it is the difference between advice
        # and silence. The reasons go on their own line only when they change, below.
        parts.append("| tracking" if belief.tracking.ready else "| LOST TRACK")
    return "  ".join(parts)


def handle(grab: Grab, reader: TableReader, tracker: RoundTracker, journal: Journal,
           store: PendingStore | None, seen: set[frozenset[str]] | None = None,
           beliefs: dict[int, RoundBelief] | None = None,
           roster_hint: tuple[str, ...] | None = None,
           advice: "LiveAdvice | None" = None) -> Record:
    """Read one frame, write one line, keep at most a few crops. The frame dies here.

    `beliefs` accumulates a `RoundBelief` per round alongside the log. It changes nothing that
    is written down -- the journal stays a record of frames, not of conclusions, so a later and
    better accumulator can be run over the same rounds rather than inheriting this one's
    mistakes. What it buys now is that a scoring event and a loss of track are announced while
    the round is still happening.
    """
    reading = reader.read(grab.pixels, grab.area, at=grab.captured_at,
                          roster_hint=roster_hint)
    round_index = tracker.observe(reading)

    crops: dict[str, str] = {}
    if store is not None and reading.screen == "table":
        stamp = time.strftime("%Y%m%dT%H%M%S", time.localtime(grab.captured_at))
        for name, box in UNREAD_REGIONS.items():
            crops[name] = store.keep(name, grab.crop(box), stamp=stamp).name

        # And one crop per distinct *failure*, not one per frame. A read that worked needs no
        # picture; the log already holds it.
        shape = refusal_shape(reading)
        if shape and seen is not None and shape not in seen:
            seen.add(shape)
            wanted = {REFUSAL_REGIONS[p] for key in shape for p in REFUSAL_REGIONS
                      if key.startswith(p)}
            note = "why-" + "-".join(sorted(shape))[:60]
            for name, box in wanted:
                crops[name] = store.keep(name, grab.crop(box), stamp=stamp, note=note).name

    belief: RoundBelief | None = None
    survey: tuple[str, ...] = ()
    if beliefs is not None:
        if round_index not in beliefs:
            beliefs[round_index] = RoundBelief(load_default())
        belief = beliefs[round_index]
        was = belief.tracking
        payout = belief.observe(reading)
        if payout is not None:
            print(f"  ** {payout}", flush=True)
        now = belief.tracking
        # Only when the verdict *changes*, so a round that is tracking says so once and a round
        # that has lost it says why once, rather than either repeating every few seconds.
        if now.ready != was.ready or now.reasons != was.reasons:
            for reason in now.reasons:
                print(f"  !! {reason}", flush=True)

        # Where the meld sits for each caller is what still blocks `scored`, and one round of
        # this produced three of the four positions. It ran only while a coin change was
        # unexplained then, and that was too narrow: 17 frames of 114, most of which caught the
        # furniture rather than a meld, and both bottom-seat calls were missed entirely.
        #
        # So it now runs on every frame that is read at all. At 54 ms against the 114 frames a
        # round produced, that is six seconds of CPU across seven minutes of play -- and the
        # ordinary frames are worth having anyway, as the baseline that tells the permanent
        # furniture apart from a meld that appeared once.
        survey = reader.survey_cards(grab.pixels, grab.area)

        # What the whole chain is for, said out loud while the round is still being played.
        # Costs 9 ms against the 285 ms read, so it runs on every frame that tracks -- but it
        # only *prints* when the recommendation changes, because a line repeated every second
        # is a line nobody reads, and the moment worth noticing is the moment it changes.
        if advice is not None:
            started = time.perf_counter()
            view, said = advice.advise(belief)
            spent = (time.perf_counter() - started) * 1000.0
            if said is not None:
                caveats = []
                if view.hand_unread:
                    caveats.append(f"{view.hand_unread} of your cards unread")
                if view.short_by:
                    caveats.append(f"{view.short_by} table cards unseen")
                short = f"  ({', '.join(caveats)})" if caveats else ""
                line = f"  >> {said.action_label}  {said.confidence:.0%}{short}"
                if line != advice.last:
                    advice.last = line
                    print(line, flush=True)
            # Sent every frame, including the ones with nothing to say: `overlay.js` renders the
            # reason when there is no hint, so a round that has lost track explains itself on the
            # panel rather than freezing on advice that is no longer true.
            if advice.overlay is not None and advice.overlay.watching:
                advice.overlay.publish(
                    hint_message(said, view, particles=advice.particles, ms=spent))

    record = Record.of(reading, round_index=round_index, crops=crops, survey=survey)
    journal.append(record)
    print(summarise(record, reading, belief), flush=True)
    return record


def _report_timing(source: MonitorSource, read: int) -> None:
    """Where a poll's time went, which decides how often a meld is seen at all.

    Worth printing every run rather than hiding behind a flag. The design assumed a poll cost
    about 40 ms and a live round measured 2.2 seconds between consecutive frames at best -- and
    a payout meld is on screen for a second or two, so that rate is the difference between
    reading `scored` and inferring it.
    """
    polls = int(source.timing["grabs"])
    if not polls:
        return
    grab = source.timing["grab"] / polls * 1000
    convert = source.timing["convert"] / polls * 1000
    print(f"timing over {polls} polls ({read} read): grab {grab:.0f} ms, "
          f"BGRA->RGB {convert:.0f} ms per poll", flush=True)
    if grab + convert > 300:
        print("  that is the bottleneck, not the reading. Tell me these numbers.", flush=True)


def tune(source: MonitorSource, seconds: float, interval: float) -> int:
    """Report what stops the table looking settled, and why a number will not read.

    Both questions need the live game and neither can be answered from a saved screenshot, so
    they share one short run. Nothing is written to disk by this at all -- not even a crop.

    The first question is which region never holds still. `SIGNATURE_REGIONS` is a guess, and
    one restless region makes the whole table look permanently in motion, which stops the gate
    firing: six records across a round where the deck fell 53 to 20.

    The second is why `coins_bottom` refuses on every live frame while reading perfectly on
    saved captures. Reported as numbers rather than a crop, because that box reaches over the
    player's name and a picture of it is a picture of a username.
    """
    from pokajan.vision.capture import region_deltas, signature
    from pokajan.vision.digits import DigitReader, describe

    digits = DigitReader.load()
    print(f"tuning for {seconds:.0f}s at {interval}s intervals -- play normally.\n"
          f"'settled' means every region moved by no more than "
          f"{__import__('pokajan.vision.capture', fromlist=['x']).SIGNATURE_TOLERANCE}.",
          flush=True)

    previous = None
    settled = polls = 0
    worst: dict[str, int] = {}
    deadline = time.time() + seconds
    last_frame: Grab | None = None

    while time.time() < deadline:
        grab = source.grab()
        if grab is None:
            time.sleep(interval)
            continue
        last_frame = grab
        current = signature(grab)
        if previous is not None:
            deltas = region_deltas(previous, current)
            polls += 1
            if deltas[0][1] <= 4:
                settled += 1
            for name, value in deltas:
                worst[name] = max(worst.get(name, 0), value)
            print("  " + "  ".join(f"{n}={v}" for n, v in deltas)
                  + ("   SETTLED" if deltas[0][1] <= 4 else ""), flush=True)
        previous = current
        time.sleep(interval)

    if not polls:
        print("never saw the game -- was it in front?")
        return 1

    print(f"\n{settled}/{polls} consecutive pairs looked settled")
    print("worst movement seen per region, largest first:")
    for name, value in sorted(worst.items(), key=lambda pair: -pair[1]):
        verdict = ("never holds still -- drop it from SIGNATURE_REGIONS" if value > 40
                   else "restless" if value > 4 else "steady")
        print(f"  {name:<18} {value:>4}   {verdict}")

    if last_frame is not None:
        print("\nwhy the coin boxes read or do not (text only -- these boxes contain a name):")
        for seat in layout.SEAT_ORDER:
            print(f"  --- coins_{seat} ---")
            for line in describe(last_frame.crop(layout.COINS[seat]), digits).splitlines():
                print(f"  {line}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--once", action="store_true", help="read one frame, then stop")
    parser.add_argument("--frame", type=Path, help="read a saved capture instead of the screen")
    parser.add_argument("--status", action="store_true", help="report what is on disk")
    parser.add_argument("--purge", action="store_true", help="delete every kept crop")
    parser.add_argument("--no-crops", action="store_true", help="keep nothing at all")
    parser.add_argument("--no-advice", action="store_true",
                        help="log only, without saying what to discard")
    parser.add_argument("--overlay", action="store_true",
                        help="also serve the overlay window's page and feed it")
    parser.add_argument("--port", type=int, default=8000,
                        help="port for --overlay (default 8000)")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="seconds between grabs (default 0.3)")
    parser.add_argument("--monitor", type=int, help="pin a monitor instead of finding one")
    parser.add_argument("--diagnose", action="store_true",
                        help="report what is on each monitor, for when the game is not found")
    parser.add_argument("--tune", type=float, metavar="SECONDS", nargs="?", const=45.0,
                        help="watch for N seconds (default 45) and report what stops the "
                             "table looking settled, plus why the coin boxes read or do not. "
                             "Writes nothing at all.")
    args = parser.parse_args(argv)

    store = PendingStore()
    if args.purge:
        removed, freed = store.purge()
        print(f"purged {removed} crops, freed {freed / 1024 / 1024:.1f} MB")
        return 0
    if args.status:
        return _status(store)

    if args.tune:
        try:
            source = MonitorSource(monitor=args.monitor)
        except ImportError:
            print("mss is not installed")
            return 2
        with source:
            print("waiting for the game -- alt-tab into it and play.", flush=True)
            while source.grab() is None:
                time.sleep(args.interval)
            return tune(source, args.tune, args.interval)

    reader = TableReader()
    tracker = RoundTracker()
    stamp = time.strftime("%Y%m%dT%H%M%S")
    journal = Journal.for_session(stamp)
    keeping = None if args.no_crops else store
    print(f"writing {journal.path.relative_to(REPO)}"
          + ("" if keeping else "  (keeping no crops)"), flush=True)

    if args.frame:
        if args.overlay:
            # Silently ignoring a flag is how someone loses ten minutes wondering why no window
            # appeared. One frame is not a round, and the overlay follows a round.
            print("--overlay does nothing with --frame: a single saved capture has no round to "
                  "follow. Run without --frame to watch the game.", flush=True)
        return _one_saved(args.frame, reader, tracker, journal, keeping)

    try:
        source = MonitorSource(monitor=args.monitor)
    except ImportError:
        print("mss is not installed -- .\\.venv\\Scripts\\pip install -r "
              "requirements-vision.txt")
        return 2

    with source:
        # Deliberately does NOT require the game to be up. It is fullscreen, so it is never in
        # front at the moment you type this command -- you are looking at a terminal. Start
        # this, alt-tab into the game, and play; the output is the log, not the console.
        print("waiting for the game -- alt-tab into it and play. Ctrl-C to stop.", flush=True)
        if args.diagnose:
            print(source.describe(), flush=True)

        counted = [0]
        ever_found = [False]
        failures: set[frozenset[str]] = set()
        beliefs: dict[int, RoundBelief] = {}
        advice = None if args.no_advice else LiveAdvice()
        if advice is not None and args.overlay:
            server = OverlayServer(port=args.port)
            if server.start():
                advice.overlay = server
                print(f"overlay served on http://127.0.0.1:{args.port}/overlay -- run "
                      f"`python -m pokajan.server.overlay` for the window", flush=True)
            else:
                # An accessory that will not start is worth one line, not a stopped watcher.
                print(f"no overlay: {server.error}", flush=True)

        def presence(here: bool) -> None:
            if here:
                ever_found[0] = True
                index = source.chosen
                monitor = source.monitors[index - 1]
                print(f"found it on monitor {index} "
                      f"({monitor['width']}x{monitor['height']}) -- watching", flush=True)
            else:
                # Only worth saying once the game has actually been seen. Before that it is
                # just the state we already announced we were waiting out -- except when there
                # is a *reason*, which there is when the picture was found and rejected rather
                # than absent, and that one must be said immediately or the whole session is
                # spent wondering why nothing appears.
                why = source.last_problem
                if why and "dark" not in why:
                    print(f"not reading: {why}", flush=True)
                elif ever_found[0]:
                    print("game not on screen (alt-tabbed, or a transition) -- still watching",
                          flush=True)

        # The turn clock is the deck counter, not stillness -- see capture.TurnGate.
        gate = TurnGate(key=deck_key(reader.digits))
        cropped_at = [0.0]
        try:
            for grab in frames(source, gate, interval=args.interval,
                               until=lambda: args.once and counted[0] >= 1,
                               on_presence=presence):
                # A payout that has been seen and not yet explained. Read at the top of the
                # loop rather than the bottom because both decisions below turn on it, and it
                # describes the frame in hand: this is the state the *previous* frames left
                # the ledger in, which is what "a payout is in flight right now" means.
                pending = any(b.ledger.pending for b in beliefs.values())

                # Crops lose to a payout, and lose to the clock the rest of the time. They
                # were costing 2 s a frame -- seven times the read they delayed -- which is
                # the real reason no five-card meld has ever been sampled. See
                # capture.crops_due.
                keep_now = keeping
                if keeping is not None and not crops_due(
                        pending=pending, at=grab.captured_at, last=cropped_at[0]):
                    keep_now = None
                elif keeping is not None:
                    cropped_at[0] = grab.captured_at

                # This round's roster, for the frames that cannot read their own. Measured
                # across a live round: the roster reads on 59% of frames and on 0% of the
                # frames showing a meld, because the payout panel that reveals the meld covers
                # the group list. Restricting the field to those ~16 holomem leaves every
                # score identical and lifts margins by up to +0.17 -- and the margin is what
                # decides a borderline card. See `TableReader.read`, which prefers what the
                # frame itself shows and only falls back to this.
                latest = beliefs[max(beliefs)] if beliefs else None
                handle(grab, reader, tracker, journal, keep_now, failures, beliefs,
                       roster_hint=latest.roster if latest else None, advice=advice)
                counted[0] += 1
                # The gate now sprints on its own, the moment the deck counter refuses -- that
                # is the payout panels covering the pile, and it arrives a full signal earlier
                # than anything here can know. See `capture.TurnGate.urgent`.
                #
                # This stays as the second half of the same idea: a payout whose panels have
                # cleared but whose coins have not been explained yet is still worth watching,
                # because the round is mid-settlement and the next call may follow immediately.
                # `TurnGate` takes whichever of the two waits is shorter, so this can only add.
                gate.heartbeat = (PAYOUT_HEARTBEAT
                                  if any(b.ledger.pending for b in beliefs.values())
                                  else IDLE_HEARTBEAT)
        except KeyboardInterrupt:
            print()
        for index in sorted(beliefs):
            belief = beliefs[index]
            print(f"round {index}: {len(belief.payouts)} payout(s), coins "
                  f"{belief.ledger.latest}, {belief.ledger.minted} minted -- "
                  f"{belief.tracking}", flush=True)
        if advice is not None and advice.overlay is not None:
            advice.overlay.stop()
        _report_timing(source, counted[0])
        if counted[0] == 0:
            print("no frames were read. If the game was up the whole time, run with "
                  "--diagnose and send me what it prints.")
            print(source.describe())
    print(f"{len(journal)} records in {journal.path.relative_to(REPO)}")
    return 0


def _one_saved(path: Path, reader, tracker, journal, store) -> int:
    """Read a screenshot from disk. For checking the pipeline without the game running."""
    from PIL import Image

    if not path.exists():
        path = REPO / "data" / "tables" / path.name
    if not path.exists():
        print(f"no such capture: {path}")
        return 2
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    area = layout.find_play_area(pixels)
    if area is None:
        print(f"{path.name}: not a game frame -- letterbox or aspect check failed")
        return 1
    handle(Grab(pixels=pixels, area=area, captured_at=time.time()),
           reader, tracker, journal, store, set(), {})
    return 0


def _status(store: PendingStore) -> int:
    files = store.files()
    print(f"kept crops: {len(files)} files, {store.total_bytes() / 1024 / 1024:.1f} MB "
          f"of {store.max_bytes / 1024 / 1024:.0f} MB cap  ({store.directory})")
    journals = sorted((store.directory.parent / "games").glob("*.jsonl"))
    if not journals:
        print("no session logs yet")
        return 0
    print(f"session logs ({len(journals)}):")
    for path in journals[-5:]:
        rounds = {r.round_index for r in Journal(path).records()}
        print(f"  {path.name}  {len(Journal(path))} records, {len(rounds)} round(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
