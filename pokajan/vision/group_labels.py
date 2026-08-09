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

So the badge alphabet has to be closed over the badges, and a whole badge is matched as one
picture rather than parsed glyph by glyph. Matching the whole thing is also what keeps
"1ID" apart from "1": the difference is entirely in the subscript, and any reader that
segments and takes the first glyph throws exactly that away.

**Aspect ratio is carried, not normalised away.** The ink is trimmed and then letterboxed
into a square canvas, so a "1" stays a narrow stripe and "My" stays wide. Stretching each
badge to fill the canvas would make those two the same shape, and the alphabet is small
enough that it cannot afford to discard a dimension that cheap.

The exemplars live in `data/captures/group_labels.yaml`, cut from real frames by
`scripts/harvest_group_labels.py` and committed as text for the same reason `digits.yaml`
is: they are shapes from the game's typeface, they carry no personal data, and committing
them is what lets the reader work on a machine with no screenshots.

Coverage is partial and the module says so. Eight of the fifteen badges have been seen; the
rest refuse. That is the intended failure, but it is a weaker guarantee than digits enjoy,
because an unseen two-glyph badge has seven seen badges to be wrong against rather than
nine well-separated numerals. `roster_panel.roster_from_groups` cross-checks every answer
against the member count the panel shows, which catches a misread whenever the true and
guessed groups differ in size -- and `LabelReader.uncovered` names what is still missing so
a capture can close the gap rather than the reader guessing through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

EXEMPLARS = Path(__file__).resolve().parents[2] / "data" / "captures" / "group_labels.yaml"

# Normalised canvas. Square, because the ink is letterboxed into it rather than stretched:
# the badge's own aspect is a feature. Bigger than a digit glyph because a badge can hold
# two shapes at different sizes and the subscript must survive the resize.
LABEL_SIZE = (28, 28)
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

    @property
    def confident(self) -> bool:
        return self.badge is not None


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


def normalise(mask: np.ndarray) -> np.ndarray:
    """Trim a badge to its ink and letterbox it into the canvas, aspect intact.

    Letterboxed, not stretched. "1" and "My" have very different proportions and that is
    most of what tells them apart once both are reduced to a small bitmap.
    """
    from PIL import Image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    canvas = np.zeros((LABEL_SIZE[1], LABEL_SIZE[0]), dtype=np.float32)
    if rows.size == 0 or cols.size == 0:
        return canvas

    trimmed = mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    height, width = trimmed.shape
    scale = min(LABEL_SIZE[0] / width, LABEL_SIZE[1] / height)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))

    picture = Image.fromarray((trimmed * 255).astype(np.uint8)).resize(target, Image.BILINEAR)
    small = np.asarray(picture, dtype=np.float32) / 255.0
    top = (LABEL_SIZE[1] - small.shape[0]) // 2
    left = (LABEL_SIZE[0] - small.shape[1]) // 2
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
    return canvas


def as_text(bitmap: np.ndarray) -> list[str]:
    return ["".join("#" if cell else "." for cell in row) for row in np.asarray(bitmap)]


def from_text(rows: Iterable[str]) -> np.ndarray:
    return np.asarray([[c == "#" for c in row] for row in rows], dtype=np.float32)


# ----------------------------------------------------------------- matching ----
class LabelReader:
    """Classifies badges against exemplars cut from the game."""

    def __init__(self, exemplars: dict[str, np.ndarray]) -> None:
        self._badges = tuple(sorted(exemplars))
        self._matrix = (
            np.stack([_unit(exemplars[b]) for b in self._badges])
            if exemplars else np.zeros((0, 1), dtype=np.float32)
        )

    @classmethod
    def load(cls, path: str | Path = EXEMPLARS) -> "LabelReader":
        path = Path(path)
        if not path.exists():
            return cls({})
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls({
            str(badge): from_text(body["rows"])
            for badge, body in (raw.get("badges") or {}).items()
        })

    def __len__(self) -> int:
        return len(self._badges)

    @property
    def badges(self) -> tuple[str, ...]:
        return self._badges

    def uncovered(self, book) -> list[str]:
        """Badges the game can print that this reader has never seen.

        Named rather than counted, because closing the gap means capturing a frame that
        shows one -- and that is only actionable if the player knows which.
        """
        return sorted({g.badge for g in book.groups} - set(self._badges))

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

        query = _unit(normalise(mask))
        scores = self._matrix @ query
        order = np.argsort(scores)[::-1]
        best = float(scores[order[0]])
        runner = self._badges[order[1]] if len(order) > 1 else ""
        margin = best - float(scores[order[1]]) if len(order) > 1 else best

        if best < MIN_SCORE:
            return Label(None, best, margin, runner,
                         f"best match {self._badges[order[0]]} only scores {best:.2f}")
        if margin < MIN_MARGIN:
            return Label(None, best, margin, runner,
                         f"{self._badges[order[0]]} at {best:.2f} with {runner} "
                         f"only {margin:.2f} behind")
        return Label(self._badges[order[0]], best, margin, runner)

    def read(self, labels: np.ndarray) -> list[Label]:
        """All four badges, top to bottom. Refusals included rather than dropped."""
        return [self.identify(band) for band in split_rows(labels)]


def _unit(bitmap: np.ndarray) -> np.ndarray:
    flat = np.asarray(bitmap, dtype=np.float32).reshape(-1)
    flat = flat - flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm else flat
