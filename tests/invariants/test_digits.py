"""Reading numbers off the screen, on synthetic images.

Same split as the card tests: the exemplars and the captures live under `data/` and are
gitignored, so accuracy against the real thing is `scripts/check_vision.py`'s job and is
reported in the README. Guarded here is the segmentation, which is where every real
failure came from -- not one was a mismatched glyph.

The failure that matters is not a refusal, it is a *plausible* wrong answer. Narrowing a
box to dodge the coin icon clipped the leading digit and turned 1430 into a confident
430, which no downstream check would have questioned.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokajan.vision.digits import (
    GLYPH_SIZE,
    DigitReader,
    Number,
    as_text,
    from_text,
    ink_mask,
    split_digits,
)

pytestmark = pytest.mark.invariant

FELT = (34, 139, 34)
INK = (250, 250, 250)


def strip(widths, *, height=40, gap=4, pad=12, extras=()):
    """A row of white blocks on felt: a stand-in for a proportional number.

    Widths differ per glyph on purpose -- the typeface is proportional, so anything that
    slices at a fixed pitch is wrong for most numbers.
    """
    total = pad * 2 + sum(widths) + gap * (len(widths) - 1)
    region = np.zeros((height + 2 * pad, total, 3), dtype=np.uint8)
    region[:, :] = FELT
    x = pad
    for width in widths:
        region[pad:pad + height, x:x + width] = INK
        x += width + gap
    for box, colour in extras:
        left, top, right, bottom = box
        region[top:bottom, left:right] = colour
    return region


# ------------------------------------------------------------ segmentation ---

def test_glyphs_are_found_as_runs_not_sliced_at_a_fixed_pitch():
    """A 1 is far narrower than a 0, so any fixed pitch cuts through most numbers."""
    assert len(split_digits(strip([8, 20, 14]))) == 3


def test_a_horizontal_rule_does_not_swallow_the_number():
    """The game underlines each seat's coin display.

    A full-width line inks every column, which collapsed whole numbers into one
    enormous glyph -- the single biggest cause of failed reads.
    """
    region = strip([14, 14, 14])
    height, width = region.shape[:2]
    region[height - 6:height - 3, :] = INK          # the underline

    assert len(split_digits(region)) == 3


def test_the_taller_band_wins_over_the_busier_one():
    """A coin box also catches the player's name on the line above.

    Choosing the band with the most ink loses numbers whenever the name is denser than
    the digits -- a name in kanji does exactly that, and the reader confidently
    segmented the name instead. The number is drawn much larger, so height decides.
    """
    region = strip([14, 14, 14], pad=30)
    # A dense but short band above: more ink than the digits, less height.
    region[4:12, 6:region.shape[1] - 6] = INK

    assert len(split_digits(region)) == 3


def test_coloured_furniture_is_not_read_as_a_digit():
    """Totals are drawn white; the coin icon beside them is yellow and the left seat's
    card backs are blue. Filtering ink by saturation is what lets the boxes stay wide
    enough for a four-digit total instead of being narrowed until they clip one."""
    plain = strip([14, 14, 14], pad=40)
    with_icon = plain.copy()
    with_icon[12:52, 2:38] = (250, 190, 40)         # a yellow disc, digit-sized

    assert len(split_digits(with_icon)) == len(split_digits(plain)) == 3


def test_two_touching_digits_are_split_at_their_pinch():
    """At the deck counter's size they routinely touch: 46, 52 and 30 each arrived as a
    single run while 71 and 63 came apart on their own.

    Wide glyphs on purpose. Two narrow ones touching are genuinely indistinguishable from
    one wide digit, and the threshold is measured from real numbers -- so the case worth
    guarding is the one where the joined run is wider than any single digit could be.
    """
    region = strip([26, 26], height=40, gap=0)      # 52 wide against 40 tall

    assert len(split_digits(region)) == 2


def test_an_empty_region_reads_as_nothing_rather_than_zero():
    felt = np.zeros((60, 200, 3), dtype=np.uint8)
    felt[:, :] = FELT

    assert split_digits(felt) == []
    assert not ink_mask(felt).any()


# ---------------------------------------------------------------- matching ---

def shape(digit: int, height: int = 40, width: int = 24) -> np.ndarray:
    """A distinctive glyph. Structured, not solid.

    A filled bar has zero variance, so a normalised correlation against it is undefined
    and every digit scores 0.00 -- which is how the first version of this fixture
    managed to make a working reader look broken.
    """
    made = np.zeros((height, width), dtype=bool)
    made[:, :] = True
    made[(digit * 3) % (height - 6):(digit * 3) % (height - 6) + 5, 2:] = False
    made[height // 2:, (digit % 3) * 6:(digit % 3) * 6 + 4] = False
    return made


def render(text: str, *, pad: int = 16, gap: int = 6) -> np.ndarray:
    """A number, drawn from the same shapes the reader is given as exemplars."""
    glyphs = [shape(int(c)) for c in text]
    height = glyphs[0].shape[0]
    total = pad * 2 + sum(g.shape[1] for g in glyphs) + gap * (len(glyphs) - 1)
    region = np.zeros((height + 2 * pad, total, 3), dtype=np.uint8)
    region[:, :] = FELT
    x = pad
    for glyph in glyphs:
        region[pad:pad + height, x:x + glyph.shape[1]][glyph] = INK
        x += glyph.shape[1] + gap
    return region


def reader() -> DigitReader:
    from pokajan.vision.digits import _normalise
    return DigitReader({str(d): _normalise(shape(d)) for d in range(10)})


@pytest.mark.parametrize("text", ["7", "10", "1430", "902"])
def test_a_number_is_read_end_to_end(text):
    found = reader().read(render(text))

    assert found.confident, found.reason
    assert found.value == int(text)


def test_one_unclear_digit_refuses_the_whole_number():
    """All-or-nothing on purpose. A coin total with one digit wrong is not approximately
    right -- it is out by a power of ten and entirely believable."""
    from pokajan.vision.digits import _normalise
    sparse = DigitReader({"7": _normalise(shape(7))})

    found = sparse.read(render("143"))

    assert not found.confident
    assert found.value is None
    assert found.reason


def test_no_exemplars_refuses_rather_than_returning_zero():
    found = DigitReader({}).read(render("4"))

    assert not found.confident
    assert "no digit exemplars" in found.reason


def test_too_many_glyphs_means_the_box_is_catching_something_else():
    found = reader().read(render("123456789"), max_digits=6)

    assert not found.confident
    assert "more than 6" in found.reason


def test_exemplars_survive_the_text_round_trip():
    """They are stored as '#' and '.' so a bad exemplar shows up in a diff."""
    glyph = shape(5)

    assert np.array_equal(from_text(as_text(glyph)) > 0.5, glyph)


def test_a_refusal_still_reports_what_it_managed():
    """So a partial read is debuggable rather than just absent."""
    found = Number(None, "14", 0.4, 0.02, "third glyph unclear")

    assert not found.confident
    assert found.digits == "14"
