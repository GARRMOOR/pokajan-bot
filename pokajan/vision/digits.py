"""Reading the numbers: coins, ranks, and the cards left in the deck.

The game draws numbers in its own typeface, so the exemplars come from the game --
`scripts/harvest_digits.py` cuts them out of captures whose values are known and writes
them to `data/captures/digits.yaml` as text. That file is committed, which is the
exception to everything else derived from the screenshots: ten digit shapes carry no
personal data, and committing them is what lets the reader run on a machine without the
captures.

Numbers are read in two steps, and the split is what makes it robust. Digits are
**found** as connected runs of ink rather than sliced at fixed widths, because the
typeface is proportional -- a 1 is far narrower than a 0 -- so any fixed pitch is wrong
for most numbers. Then each glyph is matched independently.

Ink is separated from background by brightness, not colour, and deliberately so: the
same digits appear white on green for a coin total, red for a payment and cyan for a
receipt. Matching on shape means one set of exemplars covers all of them.

A number is all-or-nothing. If any digit is unclear the whole number is refused, because
a coin total with one wrong digit is not approximately right -- it is off by a power of
ten, and it looks entirely plausible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

# Normalised glyph size. Small enough that stroke weight and antialiasing stop mattering,
# large enough to keep 3 apart from 8 and 6 apart from 5.
GLYPH_SIZE = (12, 18)          # (width, height)

EXEMPLARS = Path(__file__).resolve().parents[2] / "data" / "captures" / "digits.yaml"

# Ink is the bright part. The numbers are drawn light on a mid-green table, and the
# threshold is relative to the region's own range so it survives a darker panel.
INK_FRACTION = 0.55
# ...and the *unsaturated* part. Coin totals and the deck counter are drawn white, while
# everything that kept intruding on them is coloured: the yellow coin icon beside each
# total, and the blue card backs at the edge of the left seat's box. Filtering on
# saturation removes both wherever they sit, which is what lets the boxes stay wide enough
# for a four-digit total -- narrowing them to dodge the icon clipped the leading 1 and
# turned 1430 into a confident 430, which is exactly the silent error this module exists
# to avoid.
#
# NOTE: payment deltas are drawn in red and cyan, so reading those will need this relaxed.
MAX_INK_SATURATION = 0.35
# A run of ink columns narrower than this is a stray mark, not a digit.
MIN_GLYPH_WIDTH = 2
# A row this inked across is a rule, not glyphs. The game underlines each seat's coin
# display, and a full-width line makes every column inked -- which collapses a whole
# number into one enormous "glyph". This was the single biggest cause of failed reads.
RULE_FRACTION = 0.80
# Widest a digit may be relative to its height, and so the point at which a run is taken
# to hold two touching digits. Measured across real numbers rather than guessed: isolated
# digits in this typeface run 0.38 to 1.03, and a touching pair lands near 1.4. An earlier
# guess of 0.90 sat below the widest real digit and split every 0 in half.
MAX_GLYPH_ASPECT = 1.05
# Shortest a glyph may be relative to the tallest in the same number, which drops
# punctuation and antialiasing specks without needing to know what they are.
MIN_RELATIVE_HEIGHT = 0.55
# How sure each digit must be, and how far ahead of its runner-up.
MIN_GLYPH_SCORE = 0.55
MIN_GLYPH_MARGIN = 0.06


@dataclass(frozen=True)
class Number:
    """A number read off the screen, or a refusal."""

    value: int | None
    digits: str
    worst_score: float
    worst_margin: float
    reason: str = ""

    @property
    def confident(self) -> bool:
        return self.value is not None


# ------------------------------------------------------------ segmentation ----
def ink_mask(region: np.ndarray) -> np.ndarray:
    """Which pixels are glyph rather than background.

    Brightness, not colour, so one set of exemplars serves white totals, red payments
    and cyan receipts alike.
    """
    pixels = region.astype(np.float32)
    if pixels.ndim == 3:
        grey = pixels.mean(axis=2)
        high_channel = pixels.max(axis=2)
        low_channel = pixels.min(axis=2)
        saturation = np.where(high_channel > 0,
                              (high_channel - low_channel) / np.maximum(high_channel, 1.0),
                              0.0)
        pale = saturation <= MAX_INK_SATURATION
    else:
        grey = pixels
        pale = np.ones(grey.shape, dtype=bool)

    # The brightness range comes from the whole region, not from the pale pixels alone.
    # Taking it from the pale subset is subtly broken: where the only pale thing present
    # *is* the glyphs, the subset has no range at all, the contrast test decides the
    # region is blank, and the number vanishes. Real captures hid this behind
    # antialiasing; a clean synthetic frame exposed it immediately.
    low, high = float(grey.min()), float(grey.max())
    if high - low < 8.0:          # flat: nothing written here
        return np.zeros(grey.shape, dtype=bool)
    return pale & (grey >= low + INK_FRACTION * (high - low))


def split_digits(region: np.ndarray) -> list[np.ndarray]:
    """Cut a number into normalised glyphs, left to right.

    Columns of ink are grouped into runs, which is what makes this work on a
    proportional typeface: slicing at a fixed pitch puts the boundary through the
    middle of a digit for every number that is not all the same width.

    Three things share these regions with the number and each had to be dealt with,
    because a generous box is worth more than a hand-tuned one:

    * the **underline** beneath each seat's coin display, which inks every column and
      collapsed whole numbers into a single glyph;
    * the player's **name** on the line above, a separate band of smaller text;
    * the **coin icon** to the left of the total, a disc that reads as a wide glyph.
    """
    mask = ink_mask(region)
    if not mask.any():
        return []

    # Rules first: they would otherwise join every band and every glyph together.
    mask = mask & (mask.mean(axis=1) < RULE_FRACTION)[:, None]
    band = _busiest_band(mask)
    if band is None:
        return []
    mask = mask[band[0]:band[1]]

    runs = [
        (start, stop) for start, stop in _runs(mask.any(axis=0))
        if stop - start >= MIN_GLYPH_WIDTH
    ]
    if not runs:
        return []

    # Measured against the MEDIAN run height, not the tallest. Every digit in a number shares
    # a cap height, so the median is a digit's height whenever digits are the majority -- while
    # the tallest may be an intruder, and an intruder that is tall *raises the bar for the real
    # digits*. A live frame put two 54-56 pixel fragments of the coin icon in a box whose digits
    # are 43-45; those survived either way, but one 90-pixel intruder would have suppressed
    # every genuine digit and turned a wrong read into no read at all.
    heights = sorted(_ink_height(mask[:, a:b]) for a, b in runs)
    reference = heights[len(heights) // 2]
    kept: list[tuple[int, int]] = []
    for start, stop in runs:
        height = _ink_height(mask[:, start:stop])
        if height < MIN_RELATIVE_HEIGHT * reference:
            continue                                  # a speck, or antialiasing
        if (stop - start) <= MAX_GLYPH_ASPECT * height:
            kept.append((start, stop))
            continue
        # Too wide for one digit, so it is digits touching. At the deck counter's size
        # they routinely do -- 46, 52 and 30 each arrived as a single run while 71 and 63
        # came apart on their own.
        #
        # This is only safe because layout.py's boxes exclude the coin icon. That disc is
        # also aspect ~1 and shape cannot tell it from a touching pair, so while it was
        # inside the boxes, splitting turned it into two plausible halves and cost more
        # numbers than it recovered. The two decisions belong together.
        kept.extend(_split_wide(mask, start, stop, height))
    return [_normalise(mask[:, a:b]) for a, b in kept]


def _split_wide(mask: np.ndarray, start: int, stop: int, height: int) -> list[tuple[int, int]]:
    """Break a run holding more than one glyph, at its narrowest column.

    Touching digits still pinch where they meet, so the column carrying least ink is the
    join. Recursive, since three digits can touch as readily as two. The search skips the
    outer quarter of the run, because the thinnest column of any digit is at its own edge.
    """
    width = stop - start
    if width <= MAX_GLYPH_ASPECT * height or width < 2 * MIN_GLYPH_WIDTH:
        return [(start, stop)]

    weight = mask[:, start:stop].sum(axis=0)
    margin = max(MIN_GLYPH_WIDTH, int(round(0.25 * width)))
    middle = weight[margin:width - margin]
    if middle.size == 0:
        return [(start, stop)]
    cut = start + margin + int(np.argmin(middle))
    return (_split_wide(mask, start, cut, height)
            + _split_wide(mask, cut, stop, height))


def ink_bounds(region: np.ndarray) -> tuple[float, float, float, float] | None:
    """Where the glyphs actually are, as fractions of `region`.

    For placing the boxes in layout.py. An automated version of this -- take the union
    across every capture and pad it -- was tried and thrown away: frames where a payout
    panel covers the box put ink everywhere, so the union came back as the whole box for
    every region. Cropping one clean frame with a fraction grid drawn over it and reading
    the numbers off took two minutes and was right. This is left for spot-checking a
    single region rather than calibrating in bulk.
    """
    mask = ink_mask(region)
    if not mask.any():
        return None
    mask = mask & (mask.mean(axis=1) < RULE_FRACTION)[:, None]
    band = _busiest_band(mask)
    if band is None:
        return None
    inside = mask[band[0]:band[1]]
    cols = np.where(inside.any(axis=0))[0]
    if cols.size == 0:
        return None
    height, width = mask.shape
    return (float(cols[0]) / width, float(band[0]) / height,
            float(cols[-1] + 1) / width, float(band[1]) / height)


def describe(region: np.ndarray, reader: "DigitReader | None" = None) -> str:
    """Why a number did or did not read, as **text only**.

    Deliberately text and never a picture. The regions that need diagnosing most are the coin
    boxes, and those reach up over the player's name so that `split_digits` can find the number
    band beneath it -- so a crop of one is a picture of somebody's username, and
    `capture.CROPPABLE` refuses to save it. Numbers about the segmentation carry the same
    diagnostic weight and none of the personal data: band position, glyph count, each glyph's
    shape, and what each one nearly matched.

    Written for a specific live failure: every frame of a real session refused `coins_bottom`
    with "found 6 glyphs, more than 5", on a box that reads perfectly on saved screenshots.
    """
    mask = ink_mask(region)
    height, width = mask.shape
    lines = [f"region {width}x{height}, ink {mask.mean():.1%}"]
    if not mask.any():
        return lines[0] + " -- no ink at all"

    ruled = mask & (mask.mean(axis=1) < RULE_FRACTION)[:, None]
    rules = [band for band in _runs((mask.mean(axis=1) >= RULE_FRACTION))]
    if rules:
        lines.append(f"  full-width rules stripped at rows {rules}")

    bands = _runs(ruled.any(axis=1))
    chosen = _busiest_band(ruled)
    lines.append(f"  {len(bands)} horizontal band(s): "
                 + ", ".join(f"rows {a}-{b} (h={b - a}, ink={int(ruled[a:b].sum())})"
                             for a, b in bands))
    if chosen is None:
        return "\n".join(lines + ["  no band chosen"])
    lines.append(f"  chose rows {chosen[0]}-{chosen[1]} -- tallest, ink as tiebreak")

    inside = ruled[chosen[0]:chosen[1]]
    runs = [(a, b) for a, b in _runs(inside.any(axis=0)) if b - a >= MIN_GLYPH_WIDTH]
    tallest = max((_ink_height(inside[:, a:b]) for a, b in runs), default=0)
    for a, b in runs:
        tall = _ink_height(inside[:, a:b])
        kept = tall >= MIN_RELATIVE_HEIGHT * tallest
        wide = (b - a) > MAX_GLYPH_ASPECT * tall
        lines.append(f"    cols {a}-{b} w={b - a} h={tall} aspect={(b - a) / max(tall, 1):.2f}"
                     f"{'' if kept else '  DROPPED: too short'}"
                     f"{'  will be split: too wide' if kept and wide else ''}")

    glyphs = split_digits(region)
    lines.append(f"  -> {len(glyphs)} glyph(s) after splitting")
    if reader is not None and len(reader):
        number = reader.read(region)
        lines.append(f"  -> {number.value if number.confident else 'REFUSED'}"
                     f"  {number.reason}")
    return "\n".join(lines)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Start and stop of each consecutive True run."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, on in enumerate(list(flags) + [False]):
        if on and start is None:
            start = index
        elif not on and start is not None:
            out.append((start, index))
            start = None
    return out


def _busiest_band(mask: np.ndarray) -> tuple[int, int] | None:
    """The horizontal band of rows the number occupies.

    A coin region also catches the player's name on the line above, and the two are
    separated by blank rows. The band is chosen by **height**, with ink as a tiebreak,
    because the number is drawn considerably larger than the name -- roughly five times
    the height in these captures.

    Ink alone is not enough and picking it lost real numbers: a player whose name is
    written in kanji puts more ink on that line than "780" does on the next, so the
    reader confidently segmented the name instead.
    """
    bands = _runs(mask.any(axis=1))
    if not bands:
        return None
    return max(bands, key=lambda band: (band[1] - band[0], mask[band[0]:band[1]].sum()))


def _ink_height(glyph: np.ndarray) -> int:
    rows = np.where(glyph.any(axis=1))[0]
    return int(rows[-1] - rows[0] + 1) if rows.size else 0


def _normalise(glyph: np.ndarray) -> np.ndarray:
    """Trim to the ink and rescale to GLYPH_SIZE, as a float mask in [0, 1]."""
    rows = np.where(glyph.any(axis=1))[0]
    if rows.size == 0:
        return np.zeros((GLYPH_SIZE[1], GLYPH_SIZE[0]), dtype=np.float32)
    trimmed = glyph[rows[0]:rows[-1] + 1]

    from PIL import Image

    picture = Image.fromarray((trimmed * 255).astype(np.uint8))
    resized = picture.resize(GLYPH_SIZE, Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def as_text(glyph: np.ndarray) -> list[str]:
    """A glyph as rows of '#' and '.', for the committed exemplar file."""
    return ["".join("#" if cell else "." for cell in row) for row in np.asarray(glyph)]


def from_text(rows: Iterable[str]) -> np.ndarray:
    return np.asarray([[c == "#" for c in row] for row in rows], dtype=np.float32)


# ---------------------------------------------------------------- matching ----
class DigitReader:
    """Reads numbers using exemplars cut from the game."""

    def __init__(self, exemplars: dict[str, np.ndarray]) -> None:
        self._digits = tuple(sorted(exemplars))
        self._matrix = (
            np.stack([_unit(exemplars[d]) for d in self._digits])
            if exemplars else np.zeros((0, 1), dtype=np.float32)
        )

    @classmethod
    def load(cls, path: str | Path = EXEMPLARS) -> "DigitReader":
        path = Path(path)
        if not path.exists():
            return cls({})
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls({
            str(digit): from_text(body["rows"])
            for digit, body in (raw.get("digits") or {}).items()
        })

    def __len__(self) -> int:
        return len(self._digits)

    def read(self, region: np.ndarray, *, max_digits: int = 6) -> Number:
        """The number in this region, or a refusal.

        All-or-nothing: one unclear glyph refuses the whole number, because a coin
        total with a wrong digit is not close to right, it is out by a factor of ten and
        entirely believable.
        """
        if not len(self):
            return Number(None, "", 0.0, 0.0, "no digit exemplars loaded")

        glyphs = split_digits(region)
        if not glyphs:
            return Number(None, "", 0.0, 0.0, "nothing written here")
        if len(glyphs) > max_digits:
            return Number(None, "", 0.0, 0.0,
                          f"found {len(glyphs)} glyphs, more than {max_digits} -- "
                          f"the region is catching something else")

        found, worst_score, worst_margin = "", 1.0, 1.0
        for index, glyph in enumerate(glyphs):
            scores = self._matrix @ _unit(glyph)
            order = np.argsort(scores)[::-1]
            best = float(scores[order[0]])
            margin = best - float(scores[order[1]]) if len(order) > 1 else best
            worst_score, worst_margin = min(worst_score, best), min(worst_margin, margin)
            if best < MIN_GLYPH_SCORE or margin < MIN_GLYPH_MARGIN:
                runner = self._digits[order[1]] if len(order) > 1 else "-"
                return Number(
                    None, found, worst_score, worst_margin,
                    f"glyph {index + 1} of {len(glyphs)} is unclear: "
                    f"{self._digits[order[0]]} at {best:.2f}, "
                    f"{runner} only {margin:.2f} behind",
                )
            found += self._digits[order[0]]

        return Number(int(found), found, worst_score, worst_margin)


def _unit(glyph: np.ndarray) -> np.ndarray:
    flat = np.asarray(glyph, dtype=np.float32).reshape(-1)
    flat = flat - flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm else flat
