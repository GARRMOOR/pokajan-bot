r"""Score the card reader against real screenshots.

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
    (
        # The TOP seat's discards, which need two corrections your own row does not.
        # Their cards are drawn upside down, and they sit further up the table so they are
        # foreshortened -- 0.931 against the hand's 0.717. At the hand's aspect this row
        # segmented five cards as seven and the boundaries drifted enough to read one card
        # with its neighbour's colour. Listed in that seat's own left-to-right order, which is
        # the reverse of screen order once the row is turned upright.
        "20260809073329_1.jpg", layout.DISCARDS["top"],
        [("tsunomaki_watame", "pink"), ("tsunomaki_watame", "blue"),
         ("gawr_gura", "blue"), ("omaru_polka", "pink")],
        {"aspect": layout.DISCARD_ASPECT["top"], "rotate": 180},
    ),
    (
        "20260809073119_1.jpg", layout.DISCARDS["top"],
        [("tsunomaki_watame", "pink"), ("tsunomaki_watame", "blue"), ("gawr_gura", "blue")],
        {"aspect": layout.DISCARD_ASPECT["top"], "rotate": 180},
    ),
]


def _report_catalogue(templates: TemplateSet) -> None:
    """Whether the catalogue covers the whole agency, and every roster it can deal.

    The per-roster answer is the one that matters and it is the one `TemplateSet.missing_from`
    exists to give, because it has to be known *before* a round starts: a holomem with no art
    is a card nobody can name, and finding that out mid-hand means finding it out as a
    refusal in the middle of advising.

    Reported against every four-group combination rather than against the rosters that happen
    to be in `data/tables/`. There are only 1365 of them and the game picks freely, so
    "complete for the rounds we have screenshots of" is not the claim worth making.
    """
    import itertools

    from pokajan.core.roster import ObservedGroup, ObservedRoster
    from pokajan.vision.roster_panel import GroupBook

    book = GroupBook.load()
    names = sorted(path.stem for path in CARDS.iterdir()
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    resolved, unresolved, missing = book.coverage(names)
    print(f"  {len(set(resolved.values()))} of {len(book.characters)} holomem in "
          f"hololive_groups.yaml, from {len(names)} files")
    if unresolved:
        print(f"  FILENAMES TO FIX: {' '.join(unresolved)}")
    if missing:
        print(f"  NO ART YET: {' '.join(missing)}")

    incomplete = []
    for combination in itertools.combinations(book.groups, 4):
        roster = ObservedRoster(groups=tuple(
            ObservedGroup(id=g.id, name=g.label, members=g.members) for g in combination))
        if len({m for g in combination for m in g.members}) != len(roster.characters):
            continue                       # a holomem in two of the four; core/roster refuses
        if templates.missing_from(roster):
            incomplete.append(combination)
    total = sum(1 for _ in itertools.combinations(book.groups, 4))
    if incomplete:
        print(f"  {len(incomplete)} of {total} possible rosters have a holomem with no art")
    else:
        print(f"  every one of the {total} possible rosters is fully covered")
    print()


# (screenshot, expected meld or None). The meld is the only moment `scored` is observable, so
# what matters as much as reading the two payout frames is *not* reading one on the frames that
# show no payout -- the region holds the deck's decoy list and felt the rest of the time.
#
# The amount is not listed here. It is derived from the cards through the payout table and
# checked against the "N-Card <amount>" caption read by eye off each frame, which makes this
# two independent channels agreeing rather than one transcription.
MELDS = [
    ("20260809004802_1", [("yuzuki_choco", "blue"), ("yuzuki_choco", "pink"),
                          ("yuzuki_choco", "blue")], 120),
    ("20260809004900_1", [("aki_rosenthal", "blue"), ("aki_rosenthal", "orange"),
                          ("aki_rosenthal", "blue")], 120),
    # A bottom-seat call, which sits in a different box entirely. The colours are the whole
    # point of this case: the first box tried named the holomem correctly three times over and
    # got blue/blue/None against a true blue/pink/pink, because all three cards are the same
    # holomem and identification cannot detect a slice that has drifted by half a card.
    ("20260809004919_1", [("moona_hoshinova", "blue"), ("moona_hoshinova", "pink"),
                          ("moona_hoshinova", "pink")], 120),
    ("20260809005102_1", None, None),     # a payout, but past the point the meld is shown
    ("20260809073329_1", None, None),     # ordinary play
    ("20260809073119_1", None, None),
    ("20260809164608_1", None, None),
]


def _report_melds() -> bool:
    """Read the face-up meld where there is one, and nothing where there is not."""
    from pokajan.core.rules import load_default
    from pokajan.vision.accumulate import meld_shape
    from pokajan.vision.reader import TableReader

    reader = TableReader(cards=CARDS)
    rules = load_default()
    good = True
    for filename, expected, amount in MELDS:
        path = TABLES / f"{filename}.jpg"
        if not path.exists():
            continue
        with Image.open(path) as image:
            frame = np.asarray(image.convert("RGB"))
        area = layout.find_play_area(frame)
        if area is None:
            print(f"  {filename}: could not find the play area")
            good = False
            continue

        candidates = reader.read_meld(frame, area)
        # A shorter box fits inside a longer meld, so a group call yields a subset candidate as
        # well. Keep only those that describe a real shape -- which is what `accumulate` does.
        cards = ()
        for candidate in candidates:
            if meld_shape(candidate.cards, bonus=None) is not None:
                cards = candidate.cards
                break
        got = [(card.character, card.colour) for card in cards]
        ok = got == (expected or [])
        good &= ok
        if not cards:
            print(f"  {'ok  ' if ok else 'BAD '}{filename}: no meld"
                  + ("" if ok else f" -- expected {expected}"))
            continue

        shape = meld_shape(cards, bonus=None)
        pays = None if shape is None else rules.payout(
            shape.kind, group_size=shape.group_size, monochrome=shape.monochrome,
            bonus_copies=shape.bonus_copies)
        # The shape derived from the cards must pay what the caption said. Two channels that
        # share no machinery: one is template matching, the other is a number on the screen.
        good &= pays == amount
        print(f"  {'ok  ' if ok and pays == amount else 'BAD '}{filename}: "
              f"{' '.join(f'{who}:{col}' for who, col in got)}  -> {shape} pays {pays}"
              + ("" if pays == amount else f", caption said {amount}"))
    return good


def main() -> int:
    if not CARDS.is_dir():
        print(f"no card art at {CARDS} -- nothing to check")
        return 2

    templates = TemplateSet.load(CARDS)
    print(f"loaded art for {len(templates)} holomem")
    _report_catalogue(templates)

    total = named = coloured = refused = 0
    for case in CASES:
        filename, box, expected = case[:3]
        options = case[3] if len(case) > 3 else {}
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

        row = find_row(region, **options)
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

    print("melds (the only moment `scored` is observable):")
    melds_ok = _report_melds()
    print()

    if total:
        print(f"holomem {named}/{total}   colour {coloured}/{total}   "
              f"refused {refused}/{total}")
    return 0 if total and named == total and coloured == total and melds_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
