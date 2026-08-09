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
GROUP_PANEL = Box(0.430, 0.300, 0.565, 0.520)
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

# Scalars, for when digits are read. Each sits beside its seat's avatar.
COINS = {
    "bottom": Box(0.050, 0.760, 0.160, 0.815),
    "left": Box(0.045, 0.245, 0.140, 0.300),
    "top": Box(0.700, 0.150, 0.800, 0.205),
    "right": Box(0.885, 0.245, 0.985, 0.300),
}
RANKS = {
    "bottom": Box(0.105, 0.680, 0.165, 0.735),
    "left": Box(0.070, 0.165, 0.130, 0.220),
    "top": Box(0.725, 0.070, 0.785, 0.125),
    "right": Box(0.915, 0.165, 0.975, 0.220),
}
