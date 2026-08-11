"""Finding cards on the screen, and reading the colour off their frames.

Everything here is derived from the image rather than from a resolution, because a
table of pixel coordinates is a promise about a window size that will be broken by a
different monitor, a resized window or a UI update. The one number that *is* fixed is
the card's shape: every card the game draws has the same aspect ratio, so the row's
height calibrates its own card width and the count falls out of arithmetic.

That matters more than it sounds. The obvious way to split a row of cards is to look
for the table felt between them, and it fails: the gutters are only about a tenth
felt, and the glow the game puts around a highlighted pair erases one entirely.
Measured on a real frame, gutter detection found one card where there were seven.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mean of the supplied card art, which is all the same shape to within a few pixels.
CARD_ASPECT = 195 / 272

# The felt. Green-dominant and not bright, which no card frame is.
FELT_MARGIN = 30

# How much of a row or column must be non-felt to count as part of a card, as a
# fraction of the *most* covered row or column rather than of the region.
#
# Relative because an absolute fraction silently depends on how tightly the caller
# cropped. A discard field of four cards inside a region wide enough to hold six never
# covers 60% of it at the rounded corners, so an absolute 0.6 clipped the band top and
# bottom, shrank the estimated card height by 9%, and widened every slice enough to
# pull the next card into it. Two of four cards then fell under the match threshold.
SOLID_FRACTION = 0.5

# A stretch of felt at least this wide, as a fraction of a card, splits the row in two rather
# than being treated as a gutter between neighbours.
#
# The two are nowhere near each other in size, which is what makes this safe. A gutter is a few
# pixels; a real gap is a whole missing card, so on a live 1631-pixel hand crop the gutters
# measure about 0.05 of a card and the gap measures 1.0. Nothing observed lands between.
#
# This does not contradict the "do not split on felt" finding in this module's docstring -- it
# relies on it. Gutter detection fails because a gutter is barely felt at all; that is precisely
# why anything a third of a card wide cannot be one.
MIN_GAP = 0.35

# Frame colours, measured from real frames rather than guessed -- the mean of the most
# saturated fifth of each card's border ring, across several cards per colour:
#
#   blue    (62,130,244) (64,127,229) (76,132,229) (60,130,245)
#   orange  (235,116,49) (208,90,56)
#   pink    (247,70,175) (243,68,172) (239,74,164) (218,56,140)
#
# Recalibrate with scripts/check_vision.py if the game ever restyles its cards. The
# colour *names* come from rules/pokajan_v1.yaml; only these reference values live
# here, because they describe the artwork rather than the game.
FRAME_REFERENCES = {
    "blue": (64, 128, 233),
    "orange": (222, 103, 52),
    "pink": (240, 67, 170),
}


@dataclass(frozen=True)
class CardRow:
    """A row of cards found in one region of a frame."""

    cards: list[np.ndarray]      # each an (h, w, 3) uint8 crop
    card_width: int
    card_height: int
    origin: tuple[int, int]      # (x, y) of the row within the region searched
    runs: tuple[int, ...] = ()   # how many cards in each contiguous stretch

    @property
    def detached(self) -> int:
        """How many cards sit apart from the first stretch.

        **Structure, not meaning.** The obvious reading is the mahjong-style drawn card: a
        seat that has drawn and not yet discarded holds it apart, which would make this the
        visible form of hand size 8 and the one "this seat owes a discard" signal that survives
        a single frame, unlike the turn gate, which flashes.

        That reading is not yet earned, and two live crops are why. One shows five cards, a
        gap, then a single card of a group that does not sort next to its neighbours -- a drawn
        card, plainly. The other shows four, a gap, then two, split exactly at a group boundary,
        which the drawn-card story does not explain; it could be a sorted hand mid-animation
        with a card in flight, or the game spacing groups apart. Until a capture separates
        those, treat this as "the row is not contiguous" and nothing more. Building
        "seat has drawn" on it would turn an animation frame into a wrong turn attribution.
        """
        return sum(self.runs[1:]) if len(self.runs) > 1 else 0


def is_felt(pixels: np.ndarray) -> np.ndarray:
    """Boolean mask of table-felt pixels."""
    a = pixels.astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return (g > r + FELT_MARGIN) & (g > b + FELT_MARGIN)


def find_row(
    region: np.ndarray,
    *,
    aspect: float = CARD_ASPECT,
    expected: int | None = None,
    vertical: bool = False,
    rotate: int = 0,
) -> CardRow | None:
    """Split a region containing one row of cards into individual cards.

    `expected` cross-checks the arithmetic when the count is known from elsewhere --
    a hand size, say. It is a check and not an override: disagreeing means the row
    was not what the caller thought, and inventing the requested number of cards from
    a misread region is how a reader ends up confidently wrong.

    `vertical` for the seats either side of you, whose discards run down the screen
    rather than across it. Without it the same code returns one enormous "card" with an
    aspect of 0.42, which is not obviously wrong to anything downstream -- it is just a
    crop that never matches anything. `rotate` then turns the cards upright, in
    multiples of 90 degrees anticlockwise, so the template matcher sees what it expects.

    `aspect` is **the aspect of a card in this region of the table, which is not the same
    everywhere.** `CARD_ASPECT` is measured from the art files and holds for your own hand,
    which sits closest to the camera; everything further up the table is foreshortened by the
    perspective. Measured on live crops of lone cards: your own discards come out at 0.765
    against the hand's 0.717, and the top seat's at 0.931.

    Getting this wrong mis-counts rather than merely mis-frames, because the count is derived
    from it: the top seat's five discards came back as **seven** at the hand's aspect, and the
    boundaries then drifted far enough that one card was read with its neighbour's colour. At
    0.931 the same row gives five. Callers should pass `layout.DISCARD_ASPECT[seat]`.
    """
    if vertical:
        # Transposing costs nothing and keeps one implementation of the arithmetic.
        # A vertical column of cards is a horizontal row of cards, sideways.
        found = find_row(region.swapaxes(0, 1), aspect=aspect, expected=expected,
                         rotate=0)
        if found is None:
            return None
        return CardRow(
            cards=[_rotate(card.swapaxes(0, 1), rotate) for card in found.cards],
            card_width=found.card_height,
            card_height=found.card_width,
            origin=(found.origin[1], found.origin[0]),
            runs=found.runs,
        )

    solid = ~is_felt(region)
    by_row, by_col = solid.mean(axis=1), solid.mean(axis=0)
    if by_row.size == 0 or by_col.size == 0 or by_row.max() == 0 or by_col.max() == 0:
        return None
    rows = np.where(by_row > SOLID_FRACTION * by_row.max())[0]
    cols = np.where(by_col > SOLID_FRACTION * by_col.max())[0]
    if rows.size == 0 or cols.size == 0:
        return None

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    height = y1 - y0
    width = height * aspect
    if width < 1:
        return None

    # Slice each contiguous stretch of cards separately rather than treating the whole span as
    # one row. A hand is not always contiguous: a seat that has drawn holds the drawn card
    # detached, and a card being played leaves a hole until the row closes up. Measuring across
    # the hole made the count too large and shifted every boundary, so slices landed on felt --
    # which is why one live session refused the same hand positions on 72 of 77 frames while
    # its neighbours read perfectly. Positional, persistent, and nothing to do with the art.
    stretches = _runs(by_col > SOLID_FRACTION * by_col.max(), min_gap=max(1, round(MIN_GAP * width)))

    cards: list[np.ndarray] = []
    counts: list[int] = []
    for x0, x1 in stretches:
        count = max(1, round((x1 - x0) / width))
        step = (x1 - x0) / count
        for index in range(count):
            left = int(round(x0 + index * step))
            right = int(round(x0 + (index + 1) * step))
            cards.append(_rotate(region[y0:y1, left:right], rotate))
        counts.append(count)

    if not cards:
        return None
    if expected is not None and len(cards) != expected:
        return None

    if (rotate // 90) % 4 == 2:
        # Turning a card upright by 180 degrees also reverses the row: the top seat's leftmost
        # card is on the right of the screen. Slices come out in screen order, so they have to
        # be flipped to be in *that seat's* order.
        #
        # Not cosmetic. Discard order is information -- obs.py encodes each opponent's last
        # three, and the newest card sits at the end furthest from its seat -- so a reversed
        # list would put the oldest card where the newest belongs and be wrong in exactly the
        # way that looks right. Verified against a capture: the row reads watame, watame, gura,
        # polka upright, and came back polka, gura, watame, watame.
        cards.reverse()
        counts.reverse()

    span = stretches[0][1] - stretches[0][0]
    return CardRow(
        cards=cards,
        card_width=int(round(span / counts[0])),
        card_height=height,
        origin=(stretches[0][0], y0),
        runs=tuple(counts),
    )


def _runs(solid: np.ndarray, *, min_gap: int) -> list[tuple[int, int]]:
    """Stretches of True, merging any False gap narrower than `min_gap`."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, on in enumerate(list(solid) + [False]):
        if on and start is None:
            start = index
        elif not on and start is not None:
            spans.append((start, index))
            start = None

    merged: list[tuple[int, int]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return merged


def _rotate(card: np.ndarray, degrees: int) -> np.ndarray:
    """Turn a card upright. Multiples of 90 anticlockwise; anything else is ignored."""
    turns = (degrees // 90) % 4
    return np.rot90(card, turns) if turns else card


def frame_colour(card: np.ndarray, *, quantile: float = 0.80) -> tuple[float, float, float]:
    """The card's frame colour, as RGB.

    Sampled from the whole border ring and reduced to the most saturated fifth of it,
    which is the part that is actually frame. Sampling a single edge strip does not
    work: the artwork overflows the frame, and a card whose holomem has white hair
    reads as near-white on the left edge -- measured at (225,223,231) on a pink card,
    which classifies as nothing at all. The frame is strongly saturated and the
    overlapping artwork usually is not, so a saturation quantile separates them
    without needing to know where the overflow is.
    """
    a = card.astype(np.float32)
    h, w = a.shape[:2]
    # Skip the outermost pixels: antialiasing against the felt, plus the glow the
    # game draws around a highlighted card.
    inset_y, inset_x = max(1, int(h * 0.03)), max(1, int(w * 0.03))
    band_y, band_x = max(1, int(h * 0.08)), max(1, int(w * 0.10))

    ring = np.concatenate([
        a[inset_y:inset_y + band_y, inset_x:w - inset_x].reshape(-1, 3),
        a[h - inset_y - band_y:h - inset_y, inset_x:w - inset_x].reshape(-1, 3),
        a[inset_y:h - inset_y, inset_x:inset_x + band_x].reshape(-1, 3),
        a[inset_y:h - inset_y, w - inset_x - band_x:w - inset_x].reshape(-1, 3),
    ])
    if ring.size == 0:
        return (0.0, 0.0, 0.0)

    high, low = ring.max(axis=1), ring.min(axis=1)
    saturation = np.where(high > 0, (high - low) / np.maximum(high, 1.0), 0.0)
    keep = ring[saturation >= np.quantile(saturation, quantile)]
    return tuple(float(v) for v in keep.mean(axis=0))


def classify_colour(
    rgb: tuple[float, float, float],
    references: dict[str, tuple[int, int, int]] | None = None,
    *,
    max_distance: float = 120.0,
) -> tuple[str | None, float]:
    """Nearest reference colour, and how far away it was.

    Returns `(None, distance)` when nothing is close enough. The frame colours are
    far apart -- the nearest pair is about 190 apart in RGB -- so a sample more than
    `max_distance` from all three is a sign the crop is not a card at all rather than
    a card of an unexpected colour.
    """
    refs = references or FRAME_REFERENCES
    query = np.asarray(rgb, dtype=np.float32)
    ranked = sorted(
        (float(np.linalg.norm(query - np.asarray(ref, dtype=np.float32))), name)
        for name, ref in refs.items()
    )
    distance, name = ranked[0]
    return (name if distance <= max_distance else None), distance
