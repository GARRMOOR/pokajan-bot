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
LETTERBOX_SAMPLES = 360
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


# How far the two letterbox bars may differ before the picture is not a centred one.
#
# A fullscreen 16:9 picture is centred, so the bars match. Measured across every capture on
# disk: the worst disagreement is **1 pixel**, because letterboxing is integer arithmetic. This
# is eight times that and still catches the case it exists for by a factor of six.
#
# What it exists for is the advisor's own overlay. A monitor grab composites whatever is on
# screen, and the overlay is always-on-top by construction, so a panel overlapping a letterbox
# bar is *lit pixels outside the picture* -- and this function is a bounding box of lit pixels.
# With the overlay at its default (40, 40) the detected picture came back 49 px too tall and
# 49 px too high, an aspect of 1.7235 against 1.7778, which the +/-5% tolerance below waves
# through. Every region then lands about 24 px out at mid-screen: reads do not fail, they come
# back *wrong*, as "found 6 glyphs, more than 5" on every coin box for two entire rounds.
LETTERBOX_SKEW = 8

# Where the advisor's own overlay may sit without corrupting what the reader sees.
#
# Two separate hazards, and the second is the one that actually bit. **Inside a letterbox bar**
# the panel is lit pixels outside the picture, which stretches the bounding box `find_play_area`
# computes -- that is what `LETTERBOX_SKEW` above now refuses. **Inside a read region** the panel
# is composited over the table and read as part of it, which is worse than a misread: the reader
# would be looking at whatever the advisor last said.
#
# So the panel belongs *inside the picture* and clear of every region any reader touches, with a
# 2% margin from the picture's own edges as well -- the panel is dark, so covering the outermost
# rows would darken them and shrink the detected area from the other direction.
#
# Computed rather than eyeballed: 60.7% of the picture is claimed by some region, and this is
# the tallest band at least 900 px wide in what is left. It covers the menu button and the top
# opponent's card backs, neither of which is read or worth looking at. The largest free
# rectangle is actually bottom-right, and it is rejected on a ground geometry cannot see -- the
# game draws its own Skip and Pokajan! buttons there, and a panel over them would hide the
# controls the player needs even though clicks pass straight through.
OVERLAY_SAFE = Box(0.020, 0.020, 0.360, 0.165)


def play_area_problem(frame: np.ndarray) -> str | None:
    """Why this frame is not a game picture, or None if it is. For diagnostics."""
    return _letterbox(frame)[1]


def find_play_area(frame: np.ndarray) -> PlayArea | None:
    """Trim the letterbox bars.

    Returns None when the result is not plausibly the game -- a frame captured during
    a transition, the wrong window entirely, or the advisor's own overlay sitting in a
    letterbox bar. Refusing here is much cheaper than every region afterwards being
    silently offset, which is exactly what happened before the symmetry check existed.

    `play_area_problem` gives the reason, for when a caller has to explain itself.
    """
    return _letterbox(frame)[0]


def _letterbox(frame: np.ndarray) -> tuple[PlayArea | None, str | None]:
    if frame.ndim != 3 or frame.shape[0] < 16 or frame.shape[1] < 16:
        return None, "not an image, or far too small to be a screen"

    # Every row and every column is still tested -- the boundary is what this returns, and a
    # few pixels of error there scales into every region below. What is subsampled is the
    # pixels *within* each row and column, which only feed a 2%-lit threshold and so need a
    # few hundred samples, not a few thousand.
    #
    # This runs on every poll, whether or not the frame is worth reading, and at full
    # resolution it cost **178 ms** on a 2880x1800 grab -- more than the gate it guards and
    # enough to hold the whole watcher near half a hertz. That is the same lesson `signature`
    # learned: stride before the arithmetic, not after.
    step_x = max(1, frame.shape[1] // LETTERBOX_SAMPLES)
    step_y = max(1, frame.shape[0] // LETTERBOX_SAMPLES)
    rows = np.where((frame[:, ::step_x].max(axis=2) > LETTERBOX_MAX).mean(axis=1) > 0.02)[0]
    cols = np.where((frame[::step_y].max(axis=2) > LETTERBOX_MAX).mean(axis=0) > 0.02)[0]
    if rows.size == 0 or cols.size == 0:
        return None, "the grab is entirely dark"

    y, height = int(rows[0]), int(rows[-1]) - int(rows[0]) + 1
    x, width = int(cols[0]), int(cols[-1]) - int(cols[0]) + 1
    area = PlayArea(x=x, y=y, width=width, height=height)
    # A tolerance rather than an equality: the bars are a few pixels soft, and a JPEG
    # of them is softer still.
    if not 0.95 * PLAY_ASPECT <= area.aspect <= 1.05 * PLAY_ASPECT:
        return None, (f"the lit part of the screen is {width}x{height}, an aspect of "
                      f"{area.aspect:.3f} against 16:9's {PLAY_ASPECT:.3f}")

    # And it must be *centred*, which the aspect check alone does not give. A bar that is
    # lit at one end only stretches the box a little, and a little is enough: see
    # `LETTERBOX_SKEW`. This is the check that names the overlay.
    top, bottom = y, frame.shape[0] - (y + height)
    left, right = x, frame.shape[1] - (x + width)
    for near, far, axis in ((top, bottom, "top and bottom"), (left, right, "left and right")):
        if abs(near - far) > LETTERBOX_SKEW:
            return None, (
                f"the {axis} letterbox bars are {near} and {far} pixels, which is not a "
                f"centred picture -- something bright is sitting in a bar, and the usual "
                f"culprit is the advisor's own overlay window. Move it inside the game's "
                f"picture, clear of the black bars."
            )
    return area, None


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

# The meld a call scored, drawn face-up in the middle of the table during the payout.
#
# The only moment `scored` is observable. Those cards leave the table for good afterwards,
# and the coin displays give the *amount* -- which pins the shape to one or two candidates
# and no further, since a monochrome triple and a monochrome four-group both pay 840.
#
# **The meld is drawn toward the seat that called, so there is a box per caller.** A single box
# was an error of exactly the kind this file warns about: the two captures that pinned it agreed
# to within a pixel, and reading the coins off both showed the *same seat* had won each time.
# Two samples of one condition look exactly like two samples.
#
# The other positions came out of one round of ordinary play rather than a screenshotting
# session, via `reader.survey_cards` -- see `TABLE_INTERIOR`. What that measured is reassuring:
# every meld observed is **the same size**, 0.158 wide by 0.127 tall, which is three cards of
# 0.0527 at an aspect near 0.735. Only the position moves.
#
#   right   x 0.518-0.676  y 0.375-0.502   two captures, plus a live read of the cards
#   top     x 0.472-0.630  y 0.342-0.470   three frames of one round
#   left    x 0.323-0.481  y 0.483-0.610   one frame of one round
#   bottom  x 0.365-0.523  y 0.526-0.665   20260809004919_1, cards read blue/pink/pink
#
# Symmetry with the other three predicted `bottom` at y 0.470-0.597, and the capture puts it at
# 0.526-0.665 -- **wrong by about half a card**. That is why the prediction stayed in this
# comment instead of going in the table: a box off by half a card does not fail, it reads a
# neighbour's colour. Which is precisely what the first attempt at this box did, returning
# `moona_hoshinova` correctly three times over and the colours as blue/blue/None against a true
# blue/pink/pink. The holomem was right because all three cards were the same holomem, so
# identification could not detect the drift and only the colours gave it away.
#
# `bottom` has the one edge with no slack. Its meld sits directly above the winner's payout
# panel, so a bottom edge past about 0.671 pulls near-white panel into the frame-colour ring.
# Swept at the shared aspect, 1435 box variants read it correctly and the working region runs
# past every edge of the sweep except that one.
#
# Because the boxes overlap and a shorter box also fits inside a longer meld, `reader.read_meld`
# returns **every** candidate rather than choosing. Choosing is `accumulate`'s job, and it has
# what is needed: the ledger names the winning seat and the amount fixes the shape.
#
# It deliberately overlaps BONUS_CARD, because the meld is drawn *in front of* the bonus
# holomem. That is worth knowing in the other direction too: during a payout the bonus card
# region holds a meld card, and what has been stopping `_bonus` reading one is the
# `expected=1` cross-check -- roughly 1.8 meld cards fall inside that box, so the count
# disagrees and it refuses. Measured across three logged rounds, `bonus` never once read as
# a different holomem. Do not relax that `expected`.
# A meld is **right-anchored and grows leftward**, which one frame settled outright: the right
# seat's four-card meld ran x 0.465-0.676 and its three-card melds run 0.518-0.676. Same right
# edge, and 0.676 - 4 x 0.0527 = 0.465 to the fourth decimal.
#
# So a seat is a fixed right edge and a vertical band, and the box for a call follows from how
# many cards it scored. Widening one box to hold five and letting `find_row` count was tried
# first and does not work: the extra room reaches the deck pile and the decoy card list, which
# merge with the meld into a single stretch and take the count with them. An exactly-sized box
# has no room to catch anything else.
# The anchors are the **cards' own edges**, not a box with slack around them, because the meld
# is cut straight out of them rather than searched for. `reader.read_meld` slices `n` fixed-width
# cards leftward from the right edge and identifies each, with no segmentation step at all.
#
# That is not a simplification for its own sake -- segmentation is what was failing. `find_row`
# splits on felt, and a meld longer than three cards reaches left into the deck pile and the
# decoy card list, which merge with it into one stretch and take the count with them. A payout
# also rains coins across the table, which are orange enough to defeat colour-based separation
# too. Every group call so far has been lost that way: a four-card call read as the rightmost
# three of itself, correctly refused by `accumulate.meld_shape` as an incomplete group, and the
# fourth card never seen.
#
# Slicing needs none of that. The geometry is fully determined -- a fixed right edge, a fixed
# card width, a fixed band -- so there is nothing to detect, and what would have been a
# segmentation error becomes a card that fails to identify, which the guards already handle.
MELD_CARD = 0.0527                  # one card's width, identical in all four positions
MELD_SIZES = (3, 4, 5)              # a triple, or a group -- groups only ever have 3, 4 or 5
MELD_ANCHORS = {                    # seat -> (the fixed vertical edge, top, bottom)
    "right": (0.676, 0.375, 0.502),
    "top": (0.630, 0.342, 0.470),
    "left": (0.3229, 0.483, 0.610),
    "bottom": (0.365, 0.526, 0.665),
}

# Which edge of a meld stays put as it gets longer -- and **this is per seat, measured, and not
# a pattern to extrapolate.** Getting it wrong is silent, so the evidence for each seat is
# recorded here individually rather than as a rule.
#
# A three-card meld is the *identical three boxes* under either reading, because the two
# candidate fixed edges are exactly three cards apart. So a seat calibrated on a triple reads
# every triple perfectly and cuts every longer call out of bare felt, and nothing in the log
# tells the two apart. That cost four rounds on the bottom seat, and it cost them twice over:
# `right` and `top` were confirmed on real four-card calls, which looked like proof the rule
# was universal. Two seats agreeing is not a rule, and a length that cannot disagree is not
# evidence.
#
# `MELD_GROWS_RIGHT` is what a seat is *believed* to do. `MELD_UNSETTLED` is the seats where
# that belief has never been checked against a four- or five-card call, and for those
# `read_meld` cuts **both** directions and lets `accumulate` pick with the payout amount --
# strictly more evidence than pixels, and the same reason candidates are not chosen between by
# length either. A seat leaves `MELD_UNSETTLED` when a longer meld has actually been read there
# and confirmed by the coins.
#
#   right, top  -- grow leftward. Confirmed: real four-card group calls read and matched their
#                  amounts, and their predicted edges (0.518/0.571/0.623/0.676 and
#                  0.472/0.525/0.577/0.630) match the survey to three decimals.
#   bottom      -- grows rightward. **Confirmed by a read matched to its amount**: a
#                  monochrome five-group paying 1890 read as five distinct Myth members, all
#                  blue, with the round's bonus holomem appearing once -- and 1890 inverts to
#                  that one cell and no other. A wrong box cannot produce five distinct members
#                  of one group in the right colours. First found in the card survey, where
#                  right-anchoring needed cards at 0.260-0.365 and centring needed them from
#                  0.312 while nothing frame-coloured existed below 0.365, and content ran past
#                  the old anchor to 0.636.
#   left        -- grows rightward, and this took the longest to learn because fourteen left
#                  melds were read before one of them was longer than three cards. Settled by
#                  `left +930`, a monochrome four-group of Gen3 all in blue with the round's
#                  bonus holomem appearing once, and 930 inverts to that one cell.
#
# The direction is legible from any long meld read alongside its own three-card slice, with no
# pixels involved: growing rightward the slice is the meld's **first** three cards, growing
# leftward it is the **last** three. Across every meld ever logged that is 2 right / 0 left for
# `left`, 3 / 0 for `bottom`, 0 / 19 for `top` and 0 / 16 for `right` -- no contradictions in
# either direction. Re-run that check before trusting any change here.
#
# `MELD_UNSETTLED` is empty now and the machinery is deliberately kept: it is how a seat gets
# read at all while its direction is in doubt, and it cost nothing measurable (241 ms against
# 246 ms mean read over 18 real frames), because the second cut only fires once a triple has
# already read there.
MELD_GROWS_RIGHT = frozenset({"bottom", "left"})
MELD_UNSETTLED: frozenset[str] = frozenset()


def meld_span(seat: str) -> tuple[float, float]:
    """The x edges of this seat's *three-card* meld, which both readings agree on."""
    edge, _, _ = MELD_ANCHORS[seat]
    if seat in MELD_GROWS_RIGHT:
        return edge, edge + 3 * MELD_CARD
    return edge - 3 * MELD_CARD, edge


def meld_growths(seat: str) -> tuple[str, ...]:
    """Which growth directions are worth cutting for this seat, believed one first."""
    believed = "right" if seat in MELD_GROWS_RIGHT else "left"
    if seat not in MELD_UNSETTLED:
        return (believed,)
    return (believed, "left" if believed == "right" else "right")


def meld_cards(seat: str, cards: int, *, grow: str | None = None) -> tuple[Box, ...]:
    """One box per card of that meld, left to right."""
    if grow is None:
        grow = meld_growths(seat)[0]
    left3, right3 = meld_span(seat)
    _, top, bottom = MELD_ANCHORS[seat]
    if grow == "right":
        return tuple(Box(left3 + k * MELD_CARD, top, left3 + (k + 1) * MELD_CARD, bottom)
                     for k in range(cards))
    return tuple(Box(right3 - k * MELD_CARD, top, right3 - (k - 1) * MELD_CARD, bottom)
                 for k in range(cards, 0, -1))


def meld_box(seat: str, cards: int, *, grow: str | None = None) -> Box:
    """Where a `cards`-long meld sits when `seat` called it."""
    boxes = meld_cards(seat, cards, grow=grow)
    _, top, bottom = MELD_ANCHORS[seat]
    return Box(boxes[0].left, top, boxes[-1].right, bottom)

# No aspect constant here on purpose. A meld card is about 152x206 on a 2880x1620 play area,
# so roughly 0.74 -- but nothing needs the number, because the bands above give the height and
# `MELD_CARD` gives the width directly. An aspect is only ever needed to *derive* a count from
# a stretch of pixels, and melds are no longer counted that way.

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
# Whose turn it is, which the game draws as four bars around the central oval -- one per seat,
# on the edge nearest that seat -- with the active one lit yellow and the rest a dull green.
#
# Measured on two captures where the active seat is known: with `top` active the top bar reads
# a yellow fraction of 0.38 and the other three read **exactly 0.00**; with `left` active the
# left bar reads 0.33 and the others 0.00. Across every capture on disk the lit fractions run
# 0.14 to 0.38 and the unlit ones 0.00 to 0.01, and no frame has ever shown two lit at once.
#
# `bottom` has never been caught lit -- neither capture happens to be the player's own turn --
# but its box is placed rather than guessed: unlit it reads (45,111,17), which matches the
# other three unlit bars (43,113,12) and not the felt around it (35,123,0), whose blue channel
# is 0. So the box is on the bar. Whether it *lights* the same way is the one part of this that
# a live round still has to confirm.
TURN_INDICATORS = {
    "top": Box(0.445, 0.278, 0.548, 0.307),
    "bottom": Box(0.445, 0.530, 0.548, 0.556),
    "left": Box(0.278, 0.348, 0.305, 0.482),
    "right": Box(0.694, 0.348, 0.722, 0.482),
}

# A bar counts as lit above this share of yellow pixels. The two populations are 0.00-0.01 and
# 0.14-0.38, so this sits in the middle of a gap fourteen times wider than the noise -- and it
# is deliberately nearer the noise, because the cost of the two errors is not symmetric. A
# missed turn is a frame that says nothing; a false one names the wrong player as active.
TURN_LIT = 0.06

HAND = Box(0.163, 0.753, 0.729, 0.969)
DISCARDS = {
    "bottom": Box(0.323, 0.580, 0.660, 0.735),
    "left": Box(0.175, 0.200, 0.285, 0.620),
    "top": Box(0.360, 0.130, 0.630, 0.260),
    "right": Box(0.740, 0.200, 0.870, 0.680),
}

# The middle of the table: everything inside the four discard fields. Wide enough to hold a
# meld drawn for any of the four seats, which `PAYOUT_MELD` is not -- that box is pinned by two
# captures which turn out to be **the same caller**, so it locates a right-seat meld and nothing
# more. Where the other three seats' melds land is unknown, and so is how a four- or five-card
# row extends.
#
# This is the region `reader.survey_cards` reports the geometry of while a payout animates. Note
# what it is *not*: a crop that may be kept. The game-over banner is drawn across the middle of
# the table and reads "<player>'s score hit 0. Ending the game.", so a picture of this region is
# sometimes a picture of a username. Reporting spans as text is not a weaker version of saving
# the image -- it is the only version allowed, and it happens to be the answer anyway.
#
# The bottom edge reaches 0.70 rather than 0.64 because it has to clear the *lowest* meld, the
# bottom seat's at y 0.526-0.665. At 0.64 it clipped that one, which is part of why a round
# containing two bottom-seat calls yielded no bottom-seat geometry. The hand starts at 0.753,
# so there is still room below.
TABLE_INTERIOR = Box(0.25, 0.28, 0.80, 0.70)

# How wide a card is relative to its height *in each region*, which is not one number.
#
# `geometry.CARD_ASPECT` is 0.717, measured from the art files, and it holds for your own hand
# because that sits closest to the camera. Everything further up the table is foreshortened, so
# the same card is drawn shorter without being drawn narrower. Measured on live crops of lone
# cards: your own discards 0.765, the top seat's 0.931.
#
# This is not cosmetic, because `find_row` derives the *count* from it. At the hand's aspect the
# top seat's five discards came back as seven and the boundaries drifted enough to read one card
# with its neighbour's colour; at 0.931 the same row gives five.
#
# The two side seats have no entry on purpose. Their discards are diagonal staircases rather
# than rows, so a single aspect does not describe them and `find_row` should not be pointed at
# them at all -- see pokajan/vision/__init__.py.
DISCARD_ASPECT = {
    "bottom": 0.765,
    "top": 0.931,
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
