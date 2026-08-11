r"""Watch the game, read each settled frame, write down what was seen, keep no pictures.

    .\.venv\Scripts\python scripts\capture.py                 # watch until Ctrl-C
    .\.venv\Scripts\python scripts\capture.py --once          # one frame, then stop
    .\.venv\Scripts\python scripts\capture.py --frame <path>  # read a saved capture instead
    .\.venv\Scripts\python scripts\capture.py --status        # what is on disk right now
    .\.venv\Scripts\python scripts\capture.py --purge         # delete every kept crop

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

from pokajan.vision import layout
from pokajan.vision.capture import Grab, MonitorSource, PendingStore, TurnGate, frames
from pokajan.vision.journal import Journal, Record, RoundTracker
from pokajan.vision.reader import FrameReading, TableReader, deck_key

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


def summarise(record: Record, reading: FrameReading) -> str:
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
    return "  ".join(parts)


def handle(grab: Grab, reader: TableReader, tracker: RoundTracker, journal: Journal,
           store: PendingStore | None, seen: set[frozenset[str]] | None = None) -> Record:
    """Read one frame, write one line, keep at most a few crops. The frame dies here."""
    reading = reader.read(grab.pixels, grab.area, at=grab.captured_at)
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

    record = Record.of(reading, round_index=round_index, crops=crops)
    journal.append(record)
    print(summarise(record, reading), flush=True)
    return record


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

        def presence(here: bool) -> None:
            if here:
                ever_found[0] = True
                index = source.chosen
                monitor = source.monitors[index - 1]
                print(f"found it on monitor {index} "
                      f"({monitor['width']}x{monitor['height']}) -- watching", flush=True)
            elif ever_found[0]:
                # Only worth saying once the game has actually been seen. Before that it is
                # just the state we already announced we were waiting out.
                print("game not on screen (alt-tabbed, or a transition) -- still watching",
                      flush=True)

        # The turn clock is the deck counter, not stillness -- see capture.TurnGate.
        gate = TurnGate(key=deck_key(reader.digits))
        try:
            for grab in frames(source, gate, interval=args.interval,
                               until=lambda: args.once and counted[0] >= 1,
                               on_presence=presence):
                handle(grab, reader, tracker, journal, keeping, failures)
                counted[0] += 1
        except KeyboardInterrupt:
            print()
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
           reader, tracker, journal, store, set())
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
