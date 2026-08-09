"""Cut group-badge exemplars out of real captures and write them as text.

    .\\.venv\\Scripts\\python scripts\\harvest_group_labels.py

Same shape as `harvest_digits.py`, and for the same reason: the badges are drawn in the
game's own typeface, so the only source of exemplars is the game. This crops the four badge
bands from frames whose groups were read by eye, normalises each, and writes
`data/captures/group_labels.yaml`.

**That file is committed**, like `digits.yaml` and unlike anything image-shaped under
`data/`: it holds badge shapes and no personal data, and committing it is what lets the
reader work on a machine without the captures.

Each case also carries the member count the panel shows, which is checked against
`hololive_groups.yaml` before anything is harvested. That is not ceremony -- it is the same
cross-check the reader uses at runtime, run once over the training data, and it is what
would catch a case transcribed onto the wrong row.

The script prints the full score matrix of every exemplar against every other, because
`group_labels.MIN_MARGIN` is only meaningful next to the closest pair it has to separate.
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
from pokajan.vision.group_labels import (
    LABEL_SIZE, MIN_MARGIN, LabelReader, as_text, ink_mask, normalise, split_rows,
)
from pokajan.vision.roster_panel import GroupBook, count_members

TABLES = REPO / "data" / "tables"
OUT = REPO / "data" / "captures" / "group_labels.yaml"

# (screenshot, the four group ids top to bottom) -- all read by eye from the panel.
#
# Drawn from three different rounds so a badge is not learned from one frame: Gen5 appears
# in both roster layouts, and Gen1 in two. The frames where a payout panel covers the badges
# are deliberately absent; they are what MAX_INK exists to refuse.
CASES = [
    ("20260809073119_1", ["gamers", "gen4", "gen5", "myth"]),
    ("20260809073329_1", ["gamers", "gen4", "gen5", "myth"]),
    ("20260809001943_1", ["gen1", "gen4", "id3", "myth"]),
    ("20260809002206_1", ["gen1", "gen4", "id3", "myth"]),
    ("20260809004728_1", ["gen1", "gen2", "gen5", "id1"]),
    ("20260809004919_1", ["gen1", "gen2", "gen5", "id1"]),
]


def main() -> int:
    book = GroupBook.load()
    exemplars: dict[str, list[np.ndarray]] = defaultdict(list)

    for name, group_ids in CASES:
        path = TABLES / f"{name}.jpg"
        if not path.exists():
            print(f"skip {name}: not on this machine")
            continue
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))
        area = layout.find_play_area(frame)
        if area is None:
            print(f"skip {name}: no play area")
            continue

        # The same cross-check the reader runs live, applied to the transcription itself.
        counts = count_members(area.crop(frame, layout.GROUP_PANEL))
        expected = [book.by_id(g).size for g in group_ids]
        if counts != expected:
            print(f"skip {name}: panel counts {counts} but "
                  f"{group_ids} should be {expected} -- fix the case or the group table")
            continue

        bands = split_rows(area.crop(frame, layout.GROUP_LABELS))
        for group_id, band in zip(group_ids, bands):
            badge = book.by_id(group_id).badge
            mask = ink_mask(band)
            exemplars[badge].append(normalise(mask))
        print(f"{name}: {' '.join(book.by_id(g).badge for g in group_ids)}  counts {counts}")

    if not exemplars:
        print("\nno exemplars harvested")
        return 1

    averaged = {
        badge: np.mean(seen, axis=0) > 0.5
        for badge, seen in sorted(exemplars.items())
    }

    lines = [
        "# Group-badge exemplars, cut from real captures by",
        "# scripts/harvest_group_labels.py.",
        "#",
        "# Committed on purpose, like data/captures/digits.yaml and unlike anything",
        "# image-shaped under data/: these are shapes from the game's typeface and carry no",
        "# personal data, and committing them is what lets the reader work without the",
        "# screenshots present.",
        "#",
        "# The badge is matched whole rather than glyph by glyph. Four groups print a bare",
        "# numeral, but GAMERS prints 'Ga', Myth 'My', and the ID branches print their",
        "# generation number with a small 'ID' beneath -- and the difference between '1' and",
        "# '1ID' is entirely in that subscript, so anything that segments and reads the first",
        "# glyph confuses Gen1 with ID Gen1.",
        "#",
        "# '#' is ink, '.' is background. The ink is trimmed and letterboxed into the canvas",
        f"# with its aspect intact, so a narrow '1' stays narrow -- see",
        "# pokajan/vision/group_labels.py.",
        "",
        f"label_size: [{LABEL_SIZE[0]}, {LABEL_SIZE[1]}]",
        "badges:",
    ]
    for badge, bitmap in averaged.items():
        lines.append(f"  '{badge}':")
        lines.append(f"    seen: {len(exemplars[badge])}")
        lines.append("    rows:")
        for row in as_text(bitmap):
            lines.append(f'      - "{row}"')
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)} with {len(averaged)} badges")

    # How well separated are they? MIN_MARGIN is only meaningful next to this.
    reader = LabelReader(dict(zip(averaged, (b.astype(np.float32) for b in averaged.values()))))
    names = reader.badges
    matrix = np.stack([_unit(averaged[b]) for b in names])
    print("\nscore of each badge against every other:")
    print("        " + "".join(f"{b:>7}" for b in names))
    worst = 1.0
    for index, badge in enumerate(names):
        scores = matrix @ matrix[index]
        print(f"  {badge:>5} " + "".join(f"{s:7.2f}" for s in scores))
        others = np.delete(scores, index)
        worst = min(worst, 1.0 - float(others.max()))
    print(f"\nclosest pair leaves a margin of {worst:.2f} "
          f"(group_labels.MIN_MARGIN is {MIN_MARGIN})")

    missing = reader.uncovered(book)
    if missing:
        print(f"\nNOT YET SEEN: {' '.join(missing)} -- these refuse until a capture "
              f"shows them")
    return 0


def _unit(bitmap: np.ndarray) -> np.ndarray:
    flat = np.asarray(bitmap, dtype=np.float32).reshape(-1)
    flat = flat - flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm else flat


if __name__ == "__main__":
    raise SystemExit(main())
