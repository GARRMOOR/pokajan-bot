"""Cut digit exemplars out of real captures and write them as text.

    .\\.venv\\Scripts\\python scripts\\harvest_digits.py

The game draws numbers in its own typeface, so the only source of exemplars is the
game. This crops regions whose value is known, splits them into digits, normalises each
to a small bitmap, and writes `data/captures/digits.yaml`.

**That file is committed**, unlike everything else derived from the captures. It holds
no personal data -- ten small bitmaps of digit shapes -- and committing it is what lets
the reader work on a machine that does not have the screenshots. It is also plain text,
so a bad exemplar is visible in a diff rather than hidden in a binary.

Add cases below until all ten digits are covered; the script reports which are missing.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image

from pokajan.vision import layout
from pokajan.vision.digits import GLYPH_SIZE, as_text, split_digits

TABLES = REPO / "data" / "tables"
OUT = REPO / "data" / "captures" / "digits.yaml"

# (screenshot, box, the number actually shown) -- all read by eye.
# Deliberately drawn from several frames and several places on screen, so an exemplar
# is not overfitted to one size or one colour: coins are large and white, the deck
# counter is smaller, and both share the typeface.
CASES = [
    ("20260809073329_1.jpg", layout.COINS["bottom"], "1430"),
    ("20260809073329_1.jpg", layout.COINS["left"], "1430"),
    ("20260809073329_1.jpg", layout.COINS["top"], "10"),
    ("20260809073329_1.jpg", layout.COINS["right"], "1130"),
    ("20260809073119_1.jpg", layout.COINS["bottom"], "1780"),
    ("20260809073119_1.jpg", layout.COINS["left"], "780"),
    ("20260809073119_1.jpg", layout.COINS["top"], "420"),
    ("20260809073119_1.jpg", layout.COINS["right"], "1020"),
    ("20260809004802_1.jpg", layout.COINS["left"], "920"),
    ("20260809004802_1.jpg", layout.COINS["right"], "1380"),
    ("20260809005102_1.jpg", layout.COINS["right"], "1040"),
    # The deck counter, which is smaller and darker than a coin total -- and the only
    # place 5 and 6 appear at all, since every payout is a multiple of ten so no coin
    # total ever ends in anything else.
    ("20260809001943_1.jpg", layout.DECK_COUNTER, "71"),
    ("20260809002206_1.jpg", layout.DECK_COUNTER, "46"),
    ("20260809004728_1.jpg", layout.DECK_COUNTER, "71"),
    ("20260809004802_1.jpg", layout.DECK_COUNTER, "63"),
    ("20260809004900_1.jpg", layout.DECK_COUNTER, "52"),
    ("20260809073119_1.jpg", layout.DECK_COUNTER, "30"),
    ("20260809073329_1.jpg", layout.DECK_COUNTER, "0"),
]


def main() -> int:
    exemplars: dict[str, list[np.ndarray]] = defaultdict(list)
    for filename, box, value in CASES:
        path = TABLES / filename
        if not path.exists():
            print(f"skip {filename}: not on this machine")
            continue
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))
        area = layout.find_play_area(frame)
        if area is None:
            print(f"skip {filename}: no play area")
            continue

        glyphs = split_digits(area.crop(frame, box))
        status = "ok" if len(glyphs) == len(value) else \
            f"got {len(glyphs)} glyphs for {len(value)} digits -- SKIPPED"
        print(f"{filename} {value:>6}  {status}")
        if len(glyphs) != len(value):
            continue
        for digit, glyph in zip(value, glyphs):
            exemplars[digit].append(glyph)

    missing = sorted(set("0123456789") - set(exemplars))
    print(f"\ncovered {len(exemplars)}/10 digits")
    if missing:
        print(f"MISSING {''.join(missing)} -- add a case showing them")

    if not exemplars:
        return 1

    lines = [
        "# Digit exemplars, cut from real captures by scripts/harvest_digits.py.",
        "#",
        "# Committed on purpose, unlike everything else derived from the screenshots:",
        "# these are shapes from the game's typeface and contain no personal data, and",
        "# committing them is what lets the reader work without the captures present.",
        "#",
        "# Plain text so a bad exemplar shows up in a diff. '#' is ink, '.' is background,",
        f"# each glyph normalised to {GLYPH_SIZE[0]}x{GLYPH_SIZE[1]}.",
        "",
        f"glyph_size: [{GLYPH_SIZE[0]}, {GLYPH_SIZE[1]}]",
        "digits:",
    ]
    for digit in sorted(exemplars):
        # Averaged over every sighting, then re-thresholded: one crop can clip a stroke
        # or catch a neighbour, and the mean of several is steadier than any of them.
        stack = np.mean(exemplars[digit], axis=0)
        lines.append(f"  '{digit}':")
        lines.append(f"    seen: {len(exemplars[digit])}")
        lines.append("    rows:")
        for row in as_text(stack > 0.5):
            lines.append(f"      - \"{row}\"")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
