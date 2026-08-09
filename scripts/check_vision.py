"""Score the card reader against real screenshots.

    .\.venv\Scripts\python scripts\check_vision.py

This is a script and not a test on purpose. It needs the captures in `data/tables/`
and the art in `data/cards/`, and both are gitignored -- the screenshots are of live
online games and carry other players' usernames. So the automated suite tests the
machinery on synthetic images, and accuracy against the real thing is measured here,
by hand, on the machine that holds the data.

Ground truth is transcribed below by eye. Where it disagrees with the reader, check
the ground truth first: one of these lists was wrong on the first run, and it was not
the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from pokajan.vision import layout
from pokajan.vision.geometry import classify_colour, find_row, frame_colour
from pokajan.vision.templates import TemplateSet

CARDS = REPO / "data" / "cards"
TABLES = REPO / "data" / "tables"

# (screenshot, layout box, expected [(holomem, colour), ...]) -- read by eye.
#
# Regions come from vision/layout.py rather than being pixel coordinates here, so this
# also exercises the letterbox trim and the fractions. Only the axis-aligned rows are
# listed: the seats either side lay their discards out as sheared diagonal staircases,
# and the reader's answer to that is to target the newest card rather than segment the
# field -- see pokajan/vision/__init__.py.
CASES = [
    (
        "20260809073329_1.jpg", layout.HAND,
        [("shirakami_fubuki", "pink"), ("ookami_mio", "blue"), ("ookami_mio", "pink"),
         ("amane_kanata", "blue"), ("tokoyami_towa", "pink"),
         ("shishiro_botan", "pink"), ("mori_calliope", "blue")],
    ),
    (
        # The same seat's own discards. The leftmost card sits on a stack of buried
        # ones, so this also checks what a stacked pile does to a read.
        "20260809073329_1.jpg", layout.DISCARDS["bottom"],
        [("shishiro_botan", "orange"), ("tokoyami_towa", "blue"),
         ("amane_kanata", "blue"), ("gawr_gura", "pink")],
    ),
    (
        # The bonus holomem is drawn as a full card, so it reads with the ordinary
        # card templates and never has to be inferred from a payout.
        "20260809073329_1.jpg", layout.BONUS_CARD,
        [("gawr_gura", None)],
    ),
]


def main() -> int:
    if not CARDS.is_dir():
        print(f"no card art at {CARDS} -- nothing to check")
        return 2

    templates = TemplateSet.load(CARDS)
    print(f"loaded art for {len(templates)} holomem\n")

    total = named = coloured = refused = 0
    for filename, box, expected in CASES:
        path = TABLES / filename
        if not path.exists():
            print(f"skipping {filename}: not on this machine")
            continue

        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))

        area = layout.find_play_area(frame)
        if area is None:
            print(f"{filename}: could not find the play area")
            continue
        region = area.crop(frame, box)

        row = find_row(region)
        found = 0 if row is None else len(row.cards)
        status = "ok" if found == len(expected) else f"WRONG ({len(expected)} expected)"
        print(f"{filename} {box}\n  segmented {found} cards -- {status}")
        if row is None:
            continue

        for card, (want_who, want_colour) in zip(row.cards, expected):
            total += 1
            match = templates.identify(card)
            colour, distance = classify_colour(frame_colour(card))

            who_ok = match.character == want_who
            # The bonus card is drawn without a coloured frame, so `None` expected
            # there means "do not ask", not "should read as nothing".
            colour_ok = want_colour is None or colour == want_colour
            named += who_ok
            coloured += colour_ok
            refused += not match.confident

            flag = "ok  " if who_ok and colour_ok else "BAD "
            got = match.character or f"refused ({match.reason})"
            print(f"  {flag} {want_who:<18} {str(want_colour):<7} -> {got:<20}"
                  f" {colour or '?':<7} score {match.score:.2f}"
                  f" margin {match.margin:+.2f} colour dist {distance:.0f}")
        print()

    if total:
        print(f"holomem {named}/{total}   colour {coloured}/{total}   "
              f"refused {refused}/{total}")
    return 0 if total and named == total and coloured == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
