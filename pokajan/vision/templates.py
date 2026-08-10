"""Identifying which holomem is on a card, by matching supplied art.

The art in `data/cards/` is captured by hand, one file per holomem, and there will
never be a moment when it is complete -- the roster is redrawn every round, so a game
can always deal somebody whose card has not been seen yet. **An incomplete catalogue
is the normal state, not an error state**, and everything here is built around
refusing gracefully rather than around eventually having them all.

Two things about the art make one file per holomem enough.

It is captured "colourless", meaning the *frame* is grey while the artwork stays in
colour. The real game tints the frame and the background behind the holomem blue,
orange or pink, but the holomem is drawn identically in all three -- verified on a
frame holding the same character in two colours. So colour is read off the frame in
geometry.py and never enters the match.

And the same holomem in two groups is the same picture. Fubuki appears as both Gen1
and GAMERS, and the two files differ only in the badge and the label strip: matched
against a real card they score 0.736 and 0.728, a difference of 0.008. So group
variants are collapsed to one identity, which matters for more than tidiness -- see
`identify`.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Comparison size. Small on purpose: the art is cropped by hand, so no two templates
# agree on their margins to better than a few percent, and matching at full
# resolution mostly measures that misalignment. Scaled down, a real match scores
# 0.74-0.83 against 0.30-0.36 for the runner-up.
MATCH_SIZE = (72, 100)

# The frame is what changes between colours and between groups, so it is cropped away
# before matching. Asymmetric vertically because the bottom carries the group label.
INNER_X, INNER_TOP, INNER_BOTTOM = 0.14, 0.10, 0.22

# The query is matched at several offsets and scales, and the best one wins, because a
# card's crop is never framed quite like the hand-cut template it is compared against.
# The effect is not marginal -- on real discards it moved two cards from refused to
# named, 0.48 to 0.76 and 0.41 to 0.81, with margins going from +0.13 and +0.11 to
# +0.36 and +0.44. It is the difference between a reader that sees a discard field and
# one that shrugs at half of it.
JITTER = (-0.07, 0.0, 0.07)          # of card size, in each axis
ZOOMS = (0.90, 1.0, 1.10)
# The window the jitter grid is centred on must be *exactly* the crop above, or the
# unjittered variant is not the plain crop and the search can score a well-framed card
# lower than no search at all. It did: centring on the card's midpoint instead of the
# inner crop's cost the hand row 0.09 of score across the board while still matching.
QUERY_CENTRE_X = 0.5
QUERY_CENTRE_Y = (INNER_TOP + (1 - INNER_BOTTOM)) / 2
QUERY_HALF_W = 0.5 - INNER_X
QUERY_HALF_H = ((1 - INNER_BOTTOM) - INNER_TOP) / 2

# Refusal thresholds. The margin is the one that matters, and the absolute score is a
# weak sanity check -- see `identify` for why.
MIN_SCORE = 0.45
MIN_MARGIN = 0.12

# data/cards/shirakami_fubuki_GAMERS_COLORLESS.jpg -> shirakami_fubuki
_FILENAME = re.compile(r"^(?P<cid>.+?)(?:_[A-Z0-9]+)*_COLORLESS$")


@dataclass(frozen=True)
class Match:
    """The outcome of matching one crop against the catalogue."""

    character: str | None        # None means refused
    score: float
    margin: float                # over the best *different* holomem
    runner_up: str | None
    reason: str = ""

    @property
    def confident(self) -> bool:
        return self.character is not None


def character_from_filename(stem: str) -> str:
    """The canonical holomem id a template filename refers to.

    Two normalisations, and the second was missing for a long time without anything
    noticing.

    Group suffixes are stripped, so `shirakami_fubuki` and `shirakami_fubuki_GAMERS`
    are one identity. Uppercase-only suffixes, because holomem ids are lowercase and
    the group tags in these filenames are not.

    **And punctuation is folded away**, because the ids the rest of the program uses
    have none. Keeping `ninomae_ina'nis` intact -- which this deliberately used to do
    -- meant `identify` returned a name that was not a character in the loaded `Rules`,
    so the card could not be turned into a slot at all. It stayed invisible while Ina
    had no art: the bug arrived with the file that completed the catalogue, and showed
    up as 352 of the 1365 possible rosters reporting a holomem with no art while the
    catalogue reported 62 of 62. A read that names something the engine has never
    heard of is worse than a refusal, because a refusal is handled.
    """
    matched = _FILENAME.match(stem)
    bare = (matched.group("cid") if matched else stem).lower()
    return "".join(c for c in bare if c.isalnum() or c == "_")


class TemplateSet:
    """The card art available, keyed by holomem.

    Templates are stored flattened and normalised to zero mean and unit length, so a
    correlation is a dot product and the whole catalogue is one matrix multiply
    against the query's variants. That keeps the jittered search cheap enough not to
    need thinking about -- 27 variants against every template is a single
    (27 x D) @ (D x T) product.
    """

    def __init__(self, templates: dict[str, list[np.ndarray]]) -> None:
        self._templates = templates
        owners: list[str] = []
        rows: list[np.ndarray] = []
        for character, variants in sorted(templates.items()):
            for variant in variants:
                owners.append(character)
                rows.append(_unit(variant))
        self._owners = tuple(owners)
        self._matrix = np.stack(rows) if rows else np.zeros((0, 1), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._templates)

    @property
    def characters(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    # ------------------------------------------------------------- loading --
    @classmethod
    def load(cls, directory: str | Path) -> "TemplateSet":
        """Every image in `directory`, grouped by holomem.

        Several files may map to one holomem -- the group variants -- and all of them
        are kept. They are near-identical, so which one wins is immaterial, but
        keeping both means a card that happens to sit closer to one of them still
        matches rather than landing between the two.
        """
        directory = Path(directory)
        found: dict[str, list[np.ndarray]] = {}
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            with Image.open(path) as image:
                prepared = prepare(np.asarray(image.convert("RGB")))
            found.setdefault(character_from_filename(path.stem), []).append(prepared)
        return cls(found)

    # ------------------------------------------------------------ coverage --
    def missing_from(self, roster) -> tuple[str, ...]:
        """Which of a roster's holomem have no art.

        The reader needs this *before* a round starts, because the answer decides
        whether it can advise at all. Discovering it mid-hand means discovering it as
        a card nobody can name.
        """
        return tuple(c for c in roster.characters if c not in self._templates)

    # ------------------------------------------------------------ matching --
    def identify(self, card: np.ndarray) -> Match:
        """Which holomem this crop shows, or a refusal.

        Ranked by normalised cross-correlation, and the decision is driven by the
        **margin over the best different holomem** rather than by the top score.
        That is deliberate. A crop of somebody with no template still produces a best
        match, and it can score respectably -- the art is all portraits of anime girls
        against pale backgrounds, so a wrong answer does not look absurd to a
        correlation. What a wrong answer cannot do is stand clearly apart from the
        rest of the field: on real cards the correct holomem beat the runner-up by at
        least 0.378, while the also-rans sat within a few hundredths of each other.

        This is also why group variants are collapsed first. Left separate, Fubuki's
        two files would sit 0.008 apart at the top of the ranking, the margin would
        read as nil, and the one holomem with the most art would be the one the
        reader refused to name.
        """
        if not self._templates:
            return Match(None, 0.0, 0.0, None, "no card art loaded")

        queries = np.stack([_unit(v) for v in query_variants(card)])
        # (variants x templates) correlations, reduced to the best variant per template
        # and then to the best template per holomem.
        per_template = (queries @ self._matrix.T).max(axis=0)
        per_character: dict[str, float] = {}
        for owner, value in zip(self._owners, per_template):
            if value > per_character.get(owner, -2.0):
                per_character[owner] = float(value)

        best = sorted(((v, k) for k, v in per_character.items()), reverse=True)
        score, character = best[0]
        runner_up = best[1][1] if len(best) > 1 else None
        margin = score - best[1][0] if len(best) > 1 else score

        if score < MIN_SCORE:
            return Match(None, score, margin, runner_up,
                         f"best match {character} only scored {score:.2f}, "
                         f"under {MIN_SCORE}")
        if margin < MIN_MARGIN:
            return Match(None, score, margin, runner_up,
                         f"{character} and {runner_up} are {margin:.2f} apart, "
                         f"under {MIN_MARGIN} -- probably neither")
        return Match(character, score, margin, runner_up)


# ------------------------------------------------------------- comparison ----
def prepare(card: np.ndarray) -> np.ndarray:
    """Crop away the frame, scale to the comparison size, and drop to greyscale-ish.

    Colour is kept rather than converted to luma. Both work -- 7/7 on the frame
    tested -- but colour held a slightly wider worst-case margin (0.378 against
    0.350), and hair colour is most of what distinguishes these characters.
    """
    h, w = card.shape[:2]
    x, top, bottom = int(w * INNER_X), int(h * INNER_TOP), int(h * (1 - INNER_BOTTOM))
    inner = card[top:bottom, x:w - x]
    if inner.size == 0:
        inner = card
    image = Image.fromarray(inner.astype(np.uint8)).resize(MATCH_SIZE, Image.LANCZOS)
    return np.asarray(image, dtype=np.float32)


def query_variants(card: np.ndarray) -> list[np.ndarray]:
    """The crop, re-framed several ways, so one of them lines up with the templates.

    A card found on screen is never framed quite like a template cut by hand from a
    different screenshot, and a stacked discard pile shifts the crop further still.
    Rather than trying to segment perfectly, the match absorbs the error.
    """
    match_w, match_h = MATCH_SIZE
    out: list[np.ndarray] = []
    picture = Image.fromarray(card.astype(np.uint8))

    # One resize per zoom rather than one per variant. Cropping a window and scaling it
    # to MATCH_SIZE is the same as scaling the whole card so that the window happens to
    # be MATCH_SIZE, then cropping -- so the nine offsets at a given zoom are nine array
    # slices of one resized image. Twenty-seven Lanczos resizes per card was four fifths
    # of the reader's running time.
    for zoom in ZOOMS:
        scaled_w = int(round(match_w / (2 * QUERY_HALF_W * zoom)))
        scaled_h = int(round(match_h / (2 * QUERY_HALF_H * zoom)))
        if scaled_w < match_w or scaled_h < match_h:
            continue
        scaled = np.asarray(picture.resize((scaled_w, scaled_h), Image.LANCZOS),
                            dtype=np.float32)
        for dx, dy in itertools.product(JITTER, JITTER):
            left = int(round(scaled_w * (QUERY_CENTRE_X + dx)) - match_w / 2)
            top = int(round(scaled_h * (QUERY_CENTRE_Y + dy)) - match_h / 2)
            left = min(max(0, left), scaled_w - match_w)
            top = min(max(0, top), scaled_h - match_h)
            out.append(scaled[top:top + match_h, left:left + match_w])
    return out or [prepare(card)]


def _unit(image: np.ndarray) -> np.ndarray:
    """Flattened, mean-removed and unit length, so a dot product is a correlation."""
    flat = image.reshape(-1).astype(np.float32)
    flat = flat - flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm else flat
