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
    LABEL_SIZE, MIN_MARGIN, LabelReader, _split_badge, as_text, decompose, ink_mask,
    normalise, split_rows,
)
from pokajan.vision.roster_panel import GroupBook, count_members

TABLES = REPO / "data" / "tables"
OUT = REPO / "data" / "captures" / "group_labels.yaml"

# (screenshot, screen, the four group ids top to bottom) -- all read by eye from the panel.
#
# Two screens, and using both is deliberate rather than convenient. The reveal screen draws
# the badges about four times the size of the table's, so a badge appearing on both is
# learned across a fourfold scale change -- which is the property the normalisation claims
# and now has to earn. It also makes the reveal screen's own boxes checkable, since the
# member counts have to line up on it exactly as they do on the table.
#
# Frames where a payout panel covers the badges are deliberately absent; they are what
# MAX_INK exists to refuse.
LAYOUTS = {
    "table": ("GROUP_PANEL", "GROUP_LABELS"),
    "reveal": ("REVEAL_PANEL", "REVEAL_LABELS"),
}

CASES = [
    ("20260809073119_1", "table", ["gamers", "gen4", "gen5", "myth"]),
    ("20260809073329_1", "table", ["gamers", "gen4", "gen5", "myth"]),
    ("20260809001943_1", "table", ["gen1", "gen4", "id3", "myth"]),
    ("20260809002206_1", "table", ["gen1", "gen4", "id3", "myth"]),
    ("20260809004728_1", "table", ["gen1", "gen2", "gen5", "id1"]),
    ("20260809004919_1", "table", ["gen1", "gen2", "gen5", "id1"]),
    ("20260809004707_1", "reveal", ["gen1", "gen2", "gen5", "id1"]),
    ("20260809164608_1", "reveal", ["gen3", "holox", "id3", "advent"]),
    ("20260809170813_1", "reveal", ["gen3", "id2", "id3", "regloss"]),
    ("20260809171202_1", "reveal", ["gen3", "id2", "myth", "promise"]),
    ("20260809171550_1", "reveal", ["gen2", "gamers", "id1", "advent"]),
    ("20260809172234_1", "reveal", ["gen1", "holox", "id3", "myth"]),
    ("20260809172938_1", "reveal", ["gamers", "gen3", "gen4", "myth"]),
]

# Badges that exist in exactly one frame, on a screen that does not get a layout of its own.
#
# Gen0 appears only in 20260809173544_1, a wide shot taken as the round opens: the same
# table furniture drawn much smaller, from a camera position seen once and possibly only
# ever transient. One frame is not enough to write boxes for a third screen, and the reader
# does not need to read that shot -- the table and reveal layouts both cover the round. But
# the badge in it is a badge, and the normalisation does not care what it was cut from.
#
# (box measured by eye off the fraction grid, panel counts [5, 4, 3, 5] for Gen0, Gen4,
# ID Gen1 and Myth -- which is Gen0's size confirmed from a capture for the first time.)
ONE_OFFS = [
    ("20260809173544_1", "0", layout.Box(0.542, 0.315, 0.572, 0.360)),
]


def _load(name: str):
    path = TABLES / f"{name}.jpg"
    if not path.exists():
        print(f"skip {name}: not on this machine")
        return None
    with Image.open(path) as image:
        frame = np.asarray(image.convert("RGB"))
    area = layout.find_play_area(frame)
    if area is None:
        print(f"skip {name}: no play area")
        return None
    return frame, area


def main() -> int:
    book = GroupBook.load()
    exemplars: dict[str, list[np.ndarray]] = defaultdict(list)

    subscripts: dict[str, list[np.ndarray]] = defaultdict(list)

    def cut(badge: str, band: np.ndarray, where: str) -> None:
        """Split one band into its primary and subscript, checking the split agrees.

        The badge string says whether a subscript is expected, so a decomposition that
        disagrees is either a bad box or a wrong transcription -- and either way must not
        become an exemplar. This is the only place the geometry can be checked against
        something that knows the answer.
        """
        parts = decompose(ink_mask(band))
        expected_primary, expected_subscript = _split_badge(badge)
        if parts is None:
            print(f"  {where}: {badge} has no primary glyph -- skipped")
            return
        found = "ID" if parts.subscript is not None else ""
        if bool(found) != bool(expected_subscript):
            print(f"  {where}: {badge} decomposed with subscript={found or 'none'!r}, "
                  f"expected {expected_subscript or 'none'!r} -- skipped")
            return
        exemplars[expected_primary].append(normalise(parts.primary))
        if parts.subscript is not None:
            subscripts[expected_subscript].append(normalise(parts.subscript))

    for name, screen, group_ids in CASES:
        loaded = _load(name)
        if loaded is None:
            continue
        frame, area = loaded
        panel_box, labels_box = (getattr(layout, n) for n in LAYOUTS[screen])

        # The same cross-check the reader runs live, applied to the transcription itself.
        counts = count_members(area.crop(frame, panel_box))
        expected = [book.by_id(g).size for g in group_ids]
        if counts != expected:
            print(f"skip {name}: panel counts {counts} but "
                  f"{group_ids} should be {expected} -- fix the case or the group table")
            continue

        print(f"{name} [{screen}]: {' '.join(book.by_id(g).badge for g in group_ids)}"
              f"  counts {counts}")
        for group_id, band in zip(group_ids, split_rows(area.crop(frame, labels_box))):
            cut(book.by_id(group_id).badge, band, name)

    for name, badge, box in ONE_OFFS:
        loaded = _load(name)
        if loaded is None:
            continue
        frame, area = loaded
        print(f"{name} [one-off]: {badge}")
        cut(badge, area.crop(frame, box), name)

    if not exemplars:
        print("\nno exemplars harvested")
        return 1

    # Averaged over every sighting, then re-thresholded: one crop can clip a stroke or catch
    # a neighbour, and the mean of several is steadier than any of them. Here it also
    # averages across a fourfold scale difference between the two screens, which is the
    # normalisation being made to earn its claim rather than just assert it.
    average = lambda seen: (np.mean(seen, axis=0) > 0.5).astype(np.float32)  # noqa: E731
    primaries = {name: average(seen) for name, seen in sorted(exemplars.items())}
    subs = {name: average(seen) for name, seen in sorted(subscripts.items())}
    vocabulary = sorted({group.badge for group in book.groups})

    lines = [
        "# Group-badge exemplars, cut from real captures by",
        "# scripts/harvest_group_labels.py.",
        "#",
        "# Committed on purpose, like data/captures/digits.yaml and unlike anything",
        "# image-shaped under data/: these are shapes from the game's typeface and carry no",
        "# personal data, and committing them is what lets the reader work without the",
        "# screenshots present.",
        "#",
        "# A badge is a PRIMARY glyph plus an optional SUBSCRIPT, and the two are matched",
        "# separately. Six groups print a bare numeral, GAMERS prints 'Ga' and Myth 'My',",
        "# and the ID branches print their generation number with a small 'ID' beneath it.",
        "# Matching each badge as one picture was tried first and does not scale: '2ID' and",
        "# '3ID' share that identical 'ID' and scored 0.86 against each other, leaving 0.14",
        "# where the threshold wants 0.15 -- and since ID Gen2 and ID Gen3 both have three",
        "# members, the panel's member count cannot break the tie either. Compared as bare",
        "# numerals they score 0.74, a margin of 0.26.",
        "#",
        "# The letter badges do not split; their second letter touches the first, so each is",
        "# simply its own primary. Hence twelve primaries and one subscript.",
        "#",
        "# '#' is ink, '.' is background. Ink is trimmed and letterboxed into the canvas with",
        "# its aspect intact, so a narrow '1' stays narrow -- see",
        "# pokajan/vision/group_labels.py.",
        "",
        f"label_size: [{LABEL_SIZE[0]}, {LABEL_SIZE[1]}]",
        "",
        "# Every badge the game can print. A primary and a subscript are composed and then",
        "# checked against this, so a real primary under an unexpected subscript refuses",
        "# instead of inventing a group.",
        "badges: [" + ", ".join(f"'{b}'" for b in vocabulary) + "]",
    ]
    for key, bitmaps, counts in (("primaries", primaries, exemplars),
                                 ("subscripts", subs, subscripts)):
        lines.append("")
        lines.append(f"{key}:")
        for name, bitmap in bitmaps.items():
            lines.append(f"  '{name}':")
            lines.append(f"    seen: {len(counts[name])}")
            lines.append("    rows:")
            for row in as_text(bitmap):
                lines.append(f'      - "{row}"')
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}: {len(primaries)} primaries, "
          f"{len(subs)} subscripts, {len(vocabulary)} badges")

    # How well separated are the primaries? MIN_MARGIN is only meaningful next to this.
    names = list(primaries)
    matrix = np.stack([_unit(primaries[n]) for n in names])
    print("\nscore of each primary against every other:")
    print("      " + "".join(f"{n:>6}" for n in names))
    worst, pair = 1.0, ""
    for index, name in enumerate(names):
        scores = matrix @ matrix[index]
        print(f"  {name:>3} " + "".join(f"{s:6.2f}" for s in scores))
        rivals = scores.copy()
        rivals[index] = -2.0
        nearest = int(np.argmax(rivals))
        if 1.0 - float(rivals[nearest]) < worst:
            worst, pair = 1.0 - float(rivals[nearest]), f"{name} vs {names[nearest]}"
    print(f"\nclosest pair leaves a margin of {worst:.2f} ({pair}); "
          f"group_labels.MIN_MARGIN is {MIN_MARGIN}")
    if worst < MIN_MARGIN:
        print("  ^^ TOO CLOSE. Two badges could be confused with nothing to catch it.")

    reader = LabelReader.load()
    missing = reader.uncovered(book)
    print(f"\nreader composes {len(reader.badges)} of {len(vocabulary)} badges: "
          f"{' '.join(reader.badges)}")
    if missing:
        print(f"NOT YET SEEN: {' '.join(missing)} -- these refuse until a capture "
              f"shows them")
    return 0


def _unit(bitmap: np.ndarray) -> np.ndarray:
    flat = np.asarray(bitmap, dtype=np.float32).reshape(-1)
    flat = flat - flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm else flat


if __name__ == "__main__":
    raise SystemExit(main())
