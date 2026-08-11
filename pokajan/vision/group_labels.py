"""Reading the four group badges beside the roster panel.

This is the last piece of roster reading. `roster_panel` explains why the roster is
determined by *which four groups were drawn* rather than by recognising twenty portraits;
this module is the part that says which four.

**The badges are not digits, even though most of them look like one.** Four of the fifteen
groups are labelled with a bare numeral -- Gen0 through Gen5 -- but GAMERS prints "Ga",
Myth "My", and the ID branches print their generation number with a small "ID" beneath it:
"1ID", "2ID", "3ID". Feeding a row to `digits.DigitReader` therefore fails in the two ways
that matter:

* **"Ga" reads as a confident 0** -- 0.74, with its runner-up 0.25 behind. A G has nothing
  to compete against in a digit alphabet, so the refusal machinery never engages. Measured,
  not hypothesised.
* **"1ID" reads as a plain 1**, which is Gen1 -- a different group with different members.

So the alphabet has to be closed over the badges rather than over digits.

**A badge is read as a tall part and an optional subscript, matched separately.** The first
version matched each badge as one picture, which worked at eight badges and broke at
fifteen: "2ID" and "3ID" scored 0.86 against each other, leaving 0.14 where the threshold
wants 0.15. They share an identical "ID" and differ only in the numeral, so most of the
correlation is agreement about the part that carries no information. Raising the canvas from
28px to 96px did not move it -- 0.136 to 0.121 -- because the problem is the ratio of shared
ink to distinguishing ink, and resolution does not change a ratio.

That pair is also the worst possible one to get wrong. ID Gen2 and ID Gen3 both have three
members, so the panel's member count -- the one check that is independent of the reader --
cannot tell them apart. Nothing downstream would catch it.

Splitting them fixes it outright: the numerals alone score 0.74, a margin of 0.26. The
subscript is cleanly separable because it is a column run of its own, starting around 0.55
of the badge's width and confined to the lower half, measured consistently across a fourfold
scale change between the table and the reveal screen.

The letter badges do not split -- "Ga", "My", "Ad", "Pr", "Re" each come through as a single
run, their second letter touching the first -- so each is simply its own primary shape.
Twelve primaries then, and one subscript, and a badge is a primary plus an optional "ID".

**The subscript is matched, not merely counted.** Detecting only its presence would be
enough today, since "ID" is the only subscript that separates from its primary, but a
subscript that is present and does not look like "ID" should refuse rather than be assumed.

**Each glyph is stretched to fill the canvas, not letterboxed into it.** The first version
letterboxed, on the reasoning that a narrow "1" and a wide "My" are told apart by their
proportions. Measured, that is worse at every canvas size tried: mean pairwise score 0.31
against 0.20, tightest real margin 0.15 against 0.27. Identical padding correlates, so
preserving aspect drags every pair of badges closer together at once. Splitting the subscript
off is what made stretching safe, since aspect was only ever separating "1" from "1ID".

The exemplars live in `data/captures/group_labels.yaml`, cut from real frames by
`scripts/harvest_group_labels.py` and committed as text for the same reason `digits.yaml`
is: they are shapes from the game's typeface, they carry no personal data, and committing
them is what lets the reader work on a machine with no screenshots.

Every badge the game can print is now covered. `LabelReader.uncovered` still reports what a
given exemplar file is missing, because that is what turns a gap into a capture request
rather than a wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

EXEMPLARS = Path(__file__).resolve().parents[2] / "data" / "captures" / "group_labels.yaml"

# Normalised canvas for one glyph -- a primary or a subscript, never a whole badge. Each is
# stretched to fill it; see `normalise`, where the alternative is measured.
#
# 24x32 was picked off that same sweep: the tightest margin on a real sighting peaks there at
# 0.27, against 0.23 at digits.py's 12x18 and 0.26 at 36x36. The curve is shallow, so this is
# a plateau rather than a knife edge.
LABEL_SIZE = (24, 32)
LABEL_ROWS = 4

# Ink is bright and unsaturated, exactly as in digits.py -- the badges are drawn in the
# same near-white with a grey outline, on the same green felt.
#
# 0.45 rather than digits' 0.55, and measured rather than guessed: the faint white rules
# separating the panel's rows pass through these boxes, and at 0.45 they are still dimmer
# than the threshold in every capture, while the badge body is comfortably above it. A
# higher fraction would also work for the rules but starts eating the grey outline, which
# is where the subscript's shape lives.
INK_FRACTION = 0.45
MAX_INK_SATURATION = 0.35
# Below this the band is felt: no badge, or a payout panel covering it in flat white.
MIN_INK = 0.01
MAX_INK = 0.45

# A column run whose ink starts below this fraction of the badge's height is a subscript
# rather than part of the primary. Measured across every sighting on both screens: an "ID"
# begins at 0.48 to 0.51 of the height, and every primary begins at 0.00.
SUBSCRIPT_TOP = 0.30
# ...and carries at least this much ink relative to the primary, which keeps a speck or a
# stray antialiased pixel from being read as a subscript. Measured at 0.53 for a real one.
MIN_SUBSCRIPT_INK = 0.15
# The only subscript the game prints beneath a numeral. Kept as a name rather than assumed,
# so that a subscript which is present but does not look like this refuses.
SUBSCRIPT_NAMES = ("ID",)

# How sure a badge must be, and how far ahead of its runner-up. Both sit in a wide measured
# gap rather than being guessed at: across every capture, a badge that is really there
# scores 0.90 to 0.97 with a margin of 0.20 to 0.45, while the best score anything else
# managed -- felt, a payout panel, the reveal screen -- was 0.52. Nothing observed lies
# between 0.52 and 0.90.
#
# `scripts/harvest_group_labels.py` prints the score of every exemplar against every other
# and is the thing to re-run if these move; at eight badges the closest pair (4 and 5) still
# leaves 0.35.
MIN_SCORE = 0.70
MIN_MARGIN = 0.15


@dataclass(frozen=True)
class Label:
    """One badge read off the panel, or a refusal."""

    badge: str | None
    score: float
    margin: float
    runner_up: str = ""
    reason: str = ""
    primary: str = ""
    subscript: str = ""

    @property
    def confident(self) -> bool:
        return self.badge is not None


@dataclass(frozen=True)
class Parts:
    """A badge cut into the part that names it and the part that qualifies it."""

    primary: np.ndarray
    subscript: np.ndarray | None


class LabelError(ValueError):
    """The badges could not be read."""


# --------------------------------------------------------------- extraction ----
def ink_mask(band: np.ndarray) -> np.ndarray:
    """Which pixels are badge rather than felt.

    Brightness relative to the band's own range, intersected with "not colourful". The
    range is taken over the whole band rather than over the pale pixels alone, which is
    the bug `digits.ink_mask` documents: where the only pale thing present is the badge,
    a pale-only range has no spread, the contrast test calls the band blank, and the badge
    disappears.
    """
    pixels = np.asarray(band, dtype=np.float32)
    if pixels.ndim == 3:
        grey = pixels.mean(axis=2)
        high, low = pixels.max(axis=2), pixels.min(axis=2)
        saturation = np.where(high > 0, (high - low) / np.maximum(high, 1.0), 0.0)
        pale = saturation <= MAX_INK_SATURATION
    else:
        grey = pixels
        pale = np.ones(grey.shape, dtype=bool)

    darkest, brightest = float(grey.min()), float(grey.max())
    if brightest - darkest < 8.0:
        return np.zeros(grey.shape, dtype=bool)
    return pale & (grey >= darkest + INK_FRACTION * (brightest - darkest))


def split_rows(labels: np.ndarray) -> list[np.ndarray]:
    """The four badge bands, top to bottom.

    Equal bands rather than found ones. The panel's rows are a fixed grid and there are
    always exactly four of them, so there is nothing to discover -- and a found split would
    fail on precisely the frames that matter, where a payout panel has covered the badges
    and there is no structure to find.
    """
    height = labels.shape[0]
    if height < LABEL_ROWS:
        raise LabelError(f"label crop is too short: {height} rows")
    return [
        labels[height * row // LABEL_ROWS: height * (row + 1) // LABEL_ROWS]
        for row in range(LABEL_ROWS)
    ]


def normalise(glyph: np.ndarray) -> np.ndarray:
    """Trim a glyph to its ink and stretch it to fill the canvas.

    **Stretched, not letterboxed, and that is measured rather than assumed.** Keeping the
    aspect and padding the shorter side was tried first and is worse at every canvas size
    from 12x18 to 36x36: mean pairwise score 0.31 against 0.20, and the tightest margin on a
    real sighting 0.15 against 0.27.

    The reason is that the padding is *identical* between every pair of glyphs, so it
    correlates -- two badges that share nothing but their empty margins still agree over
    those margins, and every score is pulled toward 1 together. Discarding a real dimension
    to gain that is a bad trade twice over.

    It only became available once the subscript was split off, because separating "1" from
    "1ID" was the one thing aspect was buying. The two decisions are halves of one change.
    """
    from PIL import Image

    rows = np.where(glyph.any(axis=1))[0]
    cols = np.where(glyph.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros((LABEL_SIZE[1], LABEL_SIZE[0]), dtype=np.float32)

    trimmed = glyph[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    picture = Image.fromarray((trimmed * 255).astype(np.uint8)).resize(
        LABEL_SIZE, Image.BILINEAR)
    return np.asarray(picture, dtype=np.float32) / 255.0


def decompose(mask: np.ndarray) -> Parts | None:
    """Cut a badge into its primary glyph and its subscript, if it has one.

    The subscript is found by geometry, not by recognition, which is what lets it be
    matched independently afterwards. Every badge's ink falls into column runs; the runs
    that begin at the top of the badge are the primary, and a run that begins in the lower
    half is a subscript. Measured on both screens and at a fourfold scale difference, an
    "ID" starts at 0.48-0.51 of the height while every primary starts at 0.00, so the
    0.30 line has a wide berth on either side.

    Returns None when there is no primary at all -- all ink low, or none.
    """
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    trimmed = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    height = trimmed.shape[0]

    high: list[tuple[int, int]] = []
    low: list[tuple[int, int]] = []
    for start, stop in _runs(trimmed.any(axis=0)):
        ink_rows = np.where(trimmed[:, start:stop].any(axis=1))[0]
        top = int(ink_rows[0]) / height
        (low if top >= SUBSCRIPT_TOP else high).append((start, stop))

    if not high:
        return None
    primary = trimmed[:, high[0][0]:high[-1][1]]

    # A subscript has to carry real ink. Anything less is a speck or an antialiasing
    # fragment, and treating one of those as a subscript would turn Gen1 into ID Gen1.
    floor = MIN_SUBSCRIPT_INK * float(primary.sum())
    kept = [(a, b) for a, b in low if float(trimmed[:, a:b].sum()) >= floor]
    subscript = trimmed[:, kept[0][0]:kept[-1][1]] if kept else None
    return Parts(primary=primary, subscript=subscript)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, on in enumerate(list(flags) + [False]):
        if on and start is None:
            start = index
        elif not on and start is not None:
            out.append((start, index))
            start = None
    return out


def as_text(bitmap: np.ndarray) -> list[str]:
    return ["".join("#" if cell else "." for cell in row) for row in np.asarray(bitmap)]


def from_text(rows: Iterable[str]) -> np.ndarray:
    return np.asarray([[c == "#" for c in row] for row in rows], dtype=np.float32)


# ----------------------------------------------------------------- matching ----
class LabelReader:
    """Classifies badges against exemplars cut from the game.

    Holds two small alphabets -- twelve primary glyphs and one subscript -- and the list of
    badges the game can actually print. A read is a primary, an optional subscript, and then
    a check that the pair composes into a real badge: "Ga" with an "ID" under it is not a
    thing, and answering "GaID" would be worse than refusing.
    """

    def __init__(
        self,
        primaries: dict[str, np.ndarray],
        subscripts: dict[str, np.ndarray] | None = None,
        vocabulary: Iterable[str] = (),
    ) -> None:
        self._primaries = tuple(sorted(primaries))
        self._primary_matrix = (
            np.stack([_unit(primaries[p]) for p in self._primaries])
            if primaries else np.zeros((0, 1), dtype=np.float32)
        )
        subscripts = subscripts or {}
        self._subscripts = tuple(sorted(subscripts))
        self._subscript_matrix = (
            np.stack([_unit(subscripts[s]) for s in self._subscripts])
            if subscripts else np.zeros((0, 1), dtype=np.float32)
        )
        self._vocabulary = frozenset(vocabulary)

    @classmethod
    def load(cls, path: str | Path = EXEMPLARS) -> "LabelReader":
        path = Path(path)
        if not path.exists():
            return cls({})
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        read = lambda key: {  # noqa: E731
            str(name): from_text(body["rows"])
            for name, body in (raw.get(key) or {}).items()
        }
        return cls(read("primaries"), read("subscripts"), raw.get("badges") or ())

    def __len__(self) -> int:
        return len(self._primaries)

    @property
    def primaries(self) -> tuple[str, ...]:
        return self._primaries

    @property
    def badges(self) -> tuple[str, ...]:
        """Every badge this reader can compose, which is what a caller cares about."""
        return tuple(sorted(
            badge for badge in self._vocabulary
            if _split_badge(badge)[0] in self._primaries
            and (not _split_badge(badge)[1] or _split_badge(badge)[1] in self._subscripts)
        ))

    def uncovered(self, book) -> list[str]:
        """Badges the game can print that this reader could not compose.

        Named rather than counted, because closing the gap means capturing a frame that
        shows one -- and that is only actionable if the player knows which.
        """
        return sorted({g.badge for g in book.groups} - set(self.badges))

    def identify(self, band: np.ndarray) -> Label:
        """The badge in this band, or a refusal."""
        if not len(self):
            return Label(None, 0.0, 0.0, reason="no badge exemplars loaded")

        mask = ink_mask(band)
        ink = float(mask.mean())
        if ink < MIN_INK:
            return Label(None, 0.0, 0.0, reason="nothing written here")
        if ink > MAX_INK:
            # Flat and bright across the whole band: a payout panel is sitting over it.
            return Label(None, 0.0, 0.0,
                         reason=f"band is {ink:.0%} ink -- something is covering it")

        parts = decompose(mask)
        if parts is None:
            return Label(None, 0.0, 0.0, reason="no primary glyph in this band")

        primary, score, margin, runner = _match(
            self._primary_matrix, self._primaries, parts.primary)
        if score < MIN_SCORE:
            return Label(None, score, margin, runner, primary=primary,
                         reason=f"best match {primary} only scores {score:.2f}")
        if margin < MIN_MARGIN:
            return Label(None, score, margin, runner, primary=primary,
                         reason=f"{primary} at {score:.2f} with {runner} "
                                f"only {margin:.2f} behind")

        subscript = ""
        if parts.subscript is not None:
            subscript, sub_score, _, _ = _match(
                self._subscript_matrix, self._subscripts, parts.subscript)
            if sub_score < MIN_SCORE:
                # Present but unrecognised. Ignoring it would read ID Gen2 as Gen2.
                return Label(None, score, margin, runner, primary=primary,
                             reason=f"{primary} has a subscript that matches nothing -- "
                                    f"closest is {subscript or 'nothing'} at "
                                    f"{sub_score:.2f}")
            score, margin = min(score, sub_score), min(margin, sub_score)

        badge = primary + subscript
        if self._vocabulary and badge not in self._vocabulary:
            return Label(None, score, margin, runner, primary=primary, subscript=subscript,
                         reason=f"read {badge!r}, which no group prints")
        return Label(badge, score, margin, runner, primary=primary, subscript=subscript)

    def read(self, labels: np.ndarray) -> list[Label]:
        """All four badges, top to bottom. Refusals included rather than dropped."""
        return [self.identify(band) for band in split_rows(labels)]


def _split_badge(badge: str) -> tuple[str, str]:
    """A badge as (primary, subscript). "1ID" -> ("1", "ID"); "Ga" -> ("Ga", "")."""
    for name in SUBSCRIPT_NAMES:
        if badge.endswith(name) and len(badge) > len(name):
            return badge[:-len(name)], name
    return badge, ""


def _match(matrix: np.ndarray, names: tuple[str, ...], bitmap: np.ndarray):
    """Best name, its score, its lead over the runner-up, and who that was."""
    if not names:
        return "", 0.0, 0.0, ""
    scores = matrix @ _unit(normalise(bitmap))
    order = np.argsort(scores)[::-1]
    best = float(scores[order[0]])
    if len(order) > 1:
        return names[order[0]], best, best - float(scores[order[1]]), names[order[1]]
    return names[order[0]], best, best, ""


def _unit(bitmap: np.ndarray) -> np.ndarray:
    flat = np.asarray(bitmap, dtype=np.float32).reshape(-1)
    flat = flat - flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm else flat
