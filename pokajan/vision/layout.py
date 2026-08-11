"""Where things are on the table, as fractions rather than pixels.

Two layers, and the split is the point.

**The play area is found, not assumed.** The game renders 16:9 and letterboxes it into
whatever window it is given -- every capture so far is 2880x1800 with 90-pixel black
bars, leaving exactly 2880x1620. So the first step is trimming the bars, and everything
after that is expressed as a fraction of what is left. A table of absolute pixel
coordinates is a promise about one window size, and the promise gets broken by a
different monitor, a resized window, or the player alt-tabbing.

**Regions are generous, and what is inside them is found rather than assumed.** A
discard field grows as the round goes on, from one card to five or more, so its box has
to be big enough for the largest case -- which means it also contains felt, and
sometimes the translucent POKAJAN watermark painted on the table. `geometry.find_row`
locates the cards inside the box, so the box only has to bracket them.

Numbers here were measured off real captures and then checked by drawing them back
onto the frame, which is the only way to be sure a fraction means what it says.
`scripts/check_layout.py` regenerates that picture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Anything darker than this on every channel is letterbox rather than table.
LETTERBOX_MAX = 12
# The game's own aspect. Used as a sanity check on the trim, not to force it.
PLAY_ASPECT = 16 / 9

# Seats in play order. Turn order is clockwise, which on screen means the seat after
# you is the one on your LEFT -- see pokajan/vision/__init__.py, because getting this
# backwards resolves every claim tie to the wrong player.
SEAT_ORDER = ("bottom", "left", "top", "right")


@dataclass(frozen=True)
class Box:
    """A region of the play area, as fractions in [0, 1]."""

    left: float
    top: float
    right: float
    bottom: float

    def pixels(self, area: "PlayArea") -> tuple[int, int, int, int]:
        return (
            area.x + int(round(self.left * area.width)),
            area.y + int(round(self.top * area.height)),
            area.x + int(round(self.right * area.width)),
            area.y + int(round(self.bottom * area.height)),
        )


@dataclass(frozen=True)
class PlayArea:
    """The letterbox-trimmed game view inside a captured frame."""

    x: int
    y: int
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def crop(self, frame: np.ndarray, box: Box) -> np.ndarray:
        left, top, right, bottom = box.pixels(self)
        return frame[top:bottom, left:right]


def find_play_area(frame: np.ndarray) -> PlayArea | None:
    """Trim the letterbox bars.

    Returns None when the result is not plausibly the game -- a frame captured during
    a transition, or the wrong window entirely. Refusing here is much cheaper than
    every region afterwards being silently offset.
    """
    if frame.ndim != 3 or frame.shape[0] < 16 or frame.shape[1] < 16:
        return None

    lit = frame.max(axis=2) > LETTERBOX_MAX
    rows = np.where(lit.mean(axis=1) > 0.02)[0]
    cols = np.where(lit.mean(axis=0) > 0.02)[0]
    if rows.size == 0 or cols.size == 0:
        return None

    y, height = int(rows[0]), int(rows[-1]) - int(rows[0]) + 1
    x, width = int(cols[0]), int(cols[-1]) - int(cols[0]) + 1
    area = PlayArea(x=x, y=y, width=width, height=height)
    # A tolerance rather than an equality: the bars are a few pixels soft, and a JPEG
    # of them is softer still.
    if not 0.95 * PLAY_ASPECT <= area.aspect <= 1.05 * PLAY_ASPECT:
        return None
    return area


# --------------------------------------------------------------- the table ----
#
# Measured from data/tables/20260809073119_1.jpg and 20260809073329_1.jpg, both
# 2880x1800 with a 2880x1620 play area, then verified by drawing them back on.

# Centre furniture. All fixed, all present for the whole round. The group panel and the
# bonus card sit side by side and must not overlap: the panel's portraits are
# head-and-shoulders crops needing their own template set, while the bonus is drawn as a
# full card and reads with the ordinary card templates.
DECK = Box(0.320, 0.300, 0.425, 0.530)
# Just the number on the pile, which is the true count remaining out of 100 -- not to be
# confused with the game's "remaining cards" list, which counts a theoretical pool and
# is a decoy. See this package's __init__.
DECK_COUNTER = Box(0.350, 0.470, 0.400, 0.515)
# The 4x5 grid of member portraits, and nothing else. Tight on purpose: the row labels
# sit immediately to its right, and a box that includes them makes the grid's fifth column
# land on green felt -- which then reads as four filled rows where three have a
# placeholder, so every group-size check passes for the wrong reason.
GROUP_PANEL = Box(0.443, 0.328, 0.520, 0.497)
# The row badges: "Ga", "4", "5", "My". Reading these four is what determines the roster,
# since the game draws real hololive branches -- see vision/group_labels.py.
#
# Wider and taller than the panel beside it, and both directions are load-bearing. The ID
# branches print their generation number with a small "ID" beneath and to the right of it,
# and Myth's "y" descends: an earlier box ending at 0.560/0.497 clipped both, so "1ID" came
# back as a bare "1" -- which is Gen1, a real group with different members. A badge that
# loses its subscript does not refuse, it answers wrongly.
#
# The left edge is equally deliberate. GROUP_PANEL ends at 0.520 and the placeholder cells'
# rounded border runs to about 0.522, so a box starting at 0.520 picks up a full-height grey
# stripe down its left side -- which is a second "glyph" as far as any segmenter is
# concerned, and it moves depending on what the panel is showing. 0.524 clears it and still
# leaves room before the widest badge starts at 0.526.
#
# The right edge stops at 0.566 because the table's painted laurel decoration begins around
# 0.570 and would read as ink.
GROUP_LABELS = Box(0.524, 0.325, 0.566, 0.505)
GROUP_ROWS = 4


def group_label_row(index: int) -> Box:
    """The badge band for one panel row, counting from the top.

    The panel is a fixed grid, so the bands are arithmetic rather than found. Exposed
    because a single row is occasionally wanted on its own -- `scripts/harvest_digits.py`
    cuts the digit 5 out of a Gen5 badge, there being no coin total that ends in one.
    """
    return label_row(GROUP_LABELS, index)


def label_row(box: Box, index: int) -> Box:
    """One of the four badge bands inside a label box, whichever screen it came from."""
    if not 0 <= index < GROUP_ROWS:
        raise IndexError(f"panel has {GROUP_ROWS} rows, asked for {index}")
    span = (box.bottom - box.top) / GROUP_ROWS
    return Box(box.left, box.top + index * span, box.right, box.top + (index + 1) * span)


# ------------------------------------------------------- the reveal screen ----
#
# "Groups coming up", shown before the deal: the same four rows and the same four badges,
# drawn about four times the size and over on the left, with the bonus holomem as a large
# card on the right. Measured from data/tables/20260809164608_1.jpg and checked by drawing
# them back on.
#
# Worth reading rather than skipping past, for two reasons. It is the cleanest view of the
# roster the game ever shows -- nothing occludes it, and the portraits are large enough to
# recognise if that is ever wanted -- and it arrives *before* the first turn, so a reader
# that catches it starts the round already knowing what is in the deck rather than working
# it out from the smaller table panel.
#
# It is also a screen the table boxes must never be pointed at. They land on felt there, so
# the badges refuse -- but the table panel box happens to count [5, 5, 5, 5] on it, which is
# four perfectly legal group sizes. The badge refusal is the only thing between this screen
# and a fabricated roster, which is why the two layouts are named separately instead of one
# being tried as a fallback for the other.
#
# The screen animates in, and a mid-animation frame is at a different scale entirely. Those
# refuse on the member count, which is the intended behaviour rather than a lucky one.
REVEAL_PANEL = Box(0.039, 0.173, 0.342, 0.821)
REVEAL_LABELS = Box(0.378, 0.168, 0.490, 0.840)


def reveal_label_row(index: int) -> Box:
    return label_row(REVEAL_LABELS, index)
BONUS_CARD = Box(0.585, 0.300, 0.680, 0.520)

# Card rows. Deliberately loose: a discard field is one card wide at the deal and five
# or more later, and grows away from its seat -- so each box is sized for the largest
# case and `geometry.find_row` finds whatever is actually in it.
#
# The boxes for all four seats are verified. Reading the cards inside them is not
# equally solved: `bottom` and `top` are axis-aligned rows and segment cleanly, but the
# seats either side run as diagonal staircases -- each card offset along the row as well
# as across it, and sheared by the table's perspective. Their bounding box is therefore
# much wider than one card, so treating them as vertical columns yields aspects of 1.26
# and 1.54 where a clean sideways card would give 1.395, and the slices cut across cards
# instead of between them.
#
# Not worth a rectifier yet, because the reader has to watch continuously anyway and
# that changes the target: what it needs is the *newest* card in each field, one per
# turn, at the end furthest from its seat -- not a segmentation of the whole pile. See
# pokajan/vision/__init__.py.
HAND = Box(0.163, 0.753, 0.729, 0.969)
DISCARDS = {
    "bottom": Box(0.323, 0.580, 0.660, 0.735),
    "left": Box(0.175, 0.200, 0.285, 0.620),
    "top": Box(0.360, 0.130, 0.630, 0.260),
    "right": Box(0.740, 0.200, 0.870, 0.680),
}

# Scalars. Each still catches the player's name on the line above and the underline
# below, which `digits.split_digits` is built to cope with -- but each box now starts to
# the RIGHT of the coin icon, and that is load-bearing rather than tidy.
#
# The icon is a disc of roughly aspect 1, indistinguishable by shape from two touching
# digits. While it sat inside these boxes, splitting a too-wide run had to be forbidden,
# and forbidding it lost every number whose digits touch -- 46, 52 and 30 on the deck
# counter. Excluding the icon is what makes splitting safe, so these bounds and
# `digits.MAX_GLYPH_ASPECT` are two halves of one decision.
#
# The left seat's box also stops short of that seat's card backs. They are full-height,
# so they both bridged the name band into the number band and then passed the aspect test
# as a tall narrow "digit" -- every three-digit total there came back with four glyphs.
#
# **The "starts to the right of the icon" claim above was false, and a live capture proved
# it.** The icon was inside these boxes all along. On Steam's JPEGs it is saturated enough
# that the pale-ink filter shreds it into 8-16 pixel fragments, which the height filter then
# drops -- so it looked excluded when it was merely being tidied away. A live monitor grab of
# the same box renders it less saturated, it survives as two 54-56 pixel "glyphs", and
# `coins_bottom` refused on *every frame of a real session* with "found 6 glyphs, more than 5".
#
# The lesson is larger than these four numbers: **a threshold tuned on Steam screenshots does
# not necessarily transfer to a live grab of the same pixels.** Excluding something by geometry
# is durable; trusting a colour filter to remove it is a bet on the capture path. The left
# edges below are measured from where the leftmost digit of a four-digit total actually starts
# in a live frame -- the widest case a seat can hold, since no total can exceed about 4030.
COINS = {
    "bottom": Box(0.0825, 0.760, 0.160, 0.815),
    "left": Box(0.0555, 0.245, 0.124, 0.300),
    "top": Box(0.7015, 0.150, 0.800, 0.205),
    "right": Box(0.9090, 0.245, 0.985, 0.300),
}
RANKS = {
    "bottom": Box(0.105, 0.680, 0.165, 0.735),
    "left": Box(0.070, 0.165, 0.130, 0.220),
    "top": Box(0.725, 0.070, 0.785, 0.125),
    "right": Box(0.915, 0.165, 0.975, 0.220),
}
