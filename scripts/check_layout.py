"""Draw the layout regions back onto a real frame, so they can be checked by eye.

    .\\.venv\\Scripts\\python scripts\\check_layout.py [screenshot]

Fractions are easy to write down and impossible to verify by reading. This renders
them, which is the only way to know a box means what it says. Output goes next to the
screenshot as `<name>_layout.png` and is gitignored along with everything else under
data/.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
from PIL import Image, ImageDraw

from pokajan.vision import layout
from pokajan.vision.geometry import find_row

COLOURS = {
    "deck": (255, 220, 0),
    "groups": (0, 220, 255),
    "badges": (0, 140, 255),
    "bonus": (255, 120, 255),
    "hand": (120, 255, 120),
    "discards": (255, 140, 0),
    "coins": (255, 70, 70),
    "ranks": (170, 130, 255),
}


def _report_roster(frame: np.ndarray, area: layout.PlayArea) -> None:
    """What each layout's panel and badges say the roster is -- or why they refuse.

    Both screens are reported, not just the one the frame is of. A frame is only ever one of
    them, so the other layout's line is the wrong-screen behaviour being shown rather than
    asserted: it should refuse, and if it ever stops refusing that is visible here first.
    """
    from pokajan.vision.group_labels import LabelReader
    from pokajan.vision.roster_panel import GroupBook, PanelError, count_members, read_roster

    book, reader = GroupBook.load(), LabelReader.load()
    for screen, panel_box, labels_box in (
        ("table", layout.GROUP_PANEL, layout.GROUP_LABELS),
        ("reveal", layout.REVEAL_PANEL, layout.REVEAL_LABELS),
    ):
        print(f"\nroster, {screen} layout:")
        print(f"  member counts   {count_members(area.crop(frame, panel_box))}")
        for row, badge in enumerate(reader.read(area.crop(frame, labels_box))):
            detail = (f"{badge.badge!r} score {badge.score:.2f} margin {badge.margin:.2f}"
                      if badge.confident else f"refused -- {badge.reason}")
            print(f"  badge row {row + 1}     {detail}")
        try:
            roster = read_roster(area.crop(frame, panel_box), area.crop(frame, labels_box),
                                 book=book, reader=reader)
        except PanelError as error:
            print(f"  REFUSED: {error}")
            continue
        print(f"  {len(roster.characters)} characters: "
              + ", ".join(f"{g.name}({len(g.members)})" for g in roster.groups))


def main(argv: list[str]) -> int:
    name = argv[0] if argv else "20260809073329_1.jpg"
    path = Path(name) if Path(name).exists() else REPO / "data" / "tables" / name
    if not path.exists():
        print(f"no such capture: {path}")
        return 2

    with Image.open(path) as image:
        frame = np.asarray(image.convert("RGB"))

    area = layout.find_play_area(frame)
    if area is None:
        print("could not find the play area -- letterbox trim or aspect check failed")
        return 1
    print(f"frame {frame.shape[1]}x{frame.shape[0]}  ->  play area "
          f"{area.width}x{area.height} at ({area.x},{area.y})  aspect {area.aspect:.3f}")

    canvas = Image.fromarray(frame).convert("RGB")
    pen = ImageDraw.Draw(canvas)

    regions: list[tuple[str, str, layout.Box]] = [
        ("deck", "deck", layout.DECK),
        ("groups", "groups", layout.GROUP_PANEL),
        ("bonus", "bonus", layout.BONUS_CARD),
        ("hand", "hand", layout.HAND),
    ]
    # Each badge band separately rather than the whole label box. Both edges of that box
    # are tight against something -- the panel's placeholder border on the left, the
    # table's painted laurel on the right -- and the bands have to clear the descenders
    # on "My" and the ID branches' subscript. None of that is checkable from one outline.
    for row in range(layout.GROUP_ROWS):
        regions.append((f"badge {row}", "badges", layout.group_label_row(row)))
    # The reveal screen's own boxes, drawn on every frame so that pointing this at a
    # "Groups coming up" capture checks them as directly as the table ones.
    regions.append(("reveal groups", "groups", layout.REVEAL_PANEL))
    for row in range(layout.GROUP_ROWS):
        regions.append((f"reveal badge {row}", "badges", layout.reveal_label_row(row)))
    for seat in layout.SEAT_ORDER:
        regions.append((f"disc {seat}", "discards", layout.DISCARDS[seat]))
        regions.append((f"coins {seat}", "coins", layout.COINS[seat]))
        regions.append((f"rank {seat}", "ranks", layout.RANKS[seat]))

    for label, kind, box in regions:
        left, top, right, bottom = box.pixels(area)
        pen.rectangle([left, top, right, bottom], outline=COLOURS[kind], width=4)
        pen.text((left + 6, top + 4), label, fill=COLOURS[kind])

    # And report whether the loose card regions still segment, since that is what the
    # looseness is for.
    print("\ncard rows found inside each region:")
    for label, box in [("hand", layout.HAND)] + \
            [(f"discards {s}", layout.DISCARDS[s]) for s in layout.SEAT_ORDER]:
        row = find_row(area.crop(frame, box))
        found = "nothing" if row is None else (
            f"{len(row.cards)} cards, {row.card_width}x{row.card_height}, "
            f"aspect {row.card_width / row.card_height:.3f}")
        print(f"  {label:<18} {found}")

    _report_roster(frame, area)

    out = path.with_name(path.stem + "_layout.png")
    canvas.save(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
