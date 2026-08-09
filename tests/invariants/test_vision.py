"""The card reader's machinery, on synthetic images.

Everything here is generated. The real card art and the real screenshots both live
under `data/`, which is gitignored -- the captures are of live online games and carry
other players' usernames -- so accuracy against the real thing is measured by
`scripts/check_vision.py` on the machine that holds the data, and reported in the
README. What is guarded here is the machinery that a good template set still needs in
order to work: segmentation that calibrates itself, colour read from the frame rather
than the artwork, and refusal when the answer is not clear.

Refusal is the point of most of it. The catalogue can never be complete, because the
game redraws its roster every round and can always deal somebody whose art has not
been captured yet. A reader that names them anyway corrupts the belief silently, and
wrong advice is indistinguishable from right advice.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokajan.vision.geometry import (
    CARD_ASPECT,
    FRAME_REFERENCES,
    classify_colour,
    find_row,
    frame_colour,
)
from pokajan.vision.templates import (
    MIN_MARGIN,
    TemplateSet,
    character_from_filename,
    prepare,
    query_variants,
)

pytestmark = pytest.mark.invariant

FELT = (34, 139, 34)
CARD_H = 160
CARD_W = int(round(CARD_H * CARD_ASPECT))


def art(seed: int, height: int, width: int) -> np.ndarray:
    """A distinctive interior. Smoothed noise, so neighbouring pixels correlate the
    way a portrait's do and cross-correlation has structure to work with."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, size=(8, 6, 3)).astype(np.float32)
    ys = np.linspace(0, 7, height).astype(int)
    xs = np.linspace(0, 5, width).astype(int)
    return coarse[np.ix_(ys, xs)].astype(np.uint8)


def card(seed: int, colour: str = "pink", *, height: int = CARD_H,
         overflow: bool = False) -> np.ndarray:
    """A synthetic card: coloured frame, art inside."""
    width = int(round(height * CARD_ASPECT))
    made = np.zeros((height, width, 3), dtype=np.uint8)
    made[:, :] = FRAME_REFERENCES[colour]
    bx, by = int(width * 0.12), int(height * 0.08)
    made[by:height - by, bx:width - bx] = art(seed, height - 2 * by, width - 2 * bx)
    if overflow:
        # Pale artwork spilling over the frame, which is what defeats sampling a
        # single edge: a holomem with white hair reads as near-white there.
        made[int(height * 0.3):int(height * 0.7), :bx] = 235
    return made


def row_of(cards: list[np.ndarray], pad: int = 24) -> np.ndarray:
    """Lay cards out on felt, with a margin, the way the game does."""
    height = cards[0].shape[0]
    width = sum(c.shape[1] for c in cards)
    region = np.zeros((height + 2 * pad, width + 2 * pad, 3), dtype=np.uint8)
    region[:, :] = FELT
    x = pad
    for one in cards:
        region[pad:pad + height, x:x + one.shape[1]] = one
        x += one.shape[1]
    return region


# ------------------------------------------------------------ segmentation ---

@pytest.mark.parametrize("count", [1, 2, 4, 5, 7, 8])
def test_a_row_is_split_by_the_card_shape_not_by_the_gaps(count):
    """Gutter detection is what this replaces, and it does not survive contact.

    On a real frame the gaps between cards are only about a tenth felt, and the glow
    around a highlighted pair erases one completely -- gutter splitting found one card
    where there were seven. The aspect ratio is fixed, so the row's height calibrates
    its own card width.
    """
    region = row_of([card(i, "blue") for i in range(count)])

    row = find_row(region)

    assert row is not None
    assert len(row.cards) == count
    assert row.card_height == pytest.approx(CARD_H, abs=2)


def test_a_generous_region_does_not_change_the_answer():
    """The caller should not have to crop tightly, because a screen reader cannot.

    An absolute coverage threshold made this fail: four cards inside a region wide
    enough for six never cover 60% of it at the rounded corners, so the band was
    clipped, the card height came out 9% short, and every slice pulled in part of its
    neighbour.
    """
    cards = [card(i, "pink") for i in range(4)]
    tight = find_row(row_of(cards, pad=10))
    loose = find_row(row_of(cards, pad=220))

    assert tight is not None and loose is not None
    assert len(loose.cards) == len(tight.cards) == 4
    assert loose.card_height == pytest.approx(tight.card_height, abs=2)


def test_disagreeing_with_an_expected_count_refuses_rather_than_inventing():
    """A hand size known from elsewhere is a cross-check, never an override.

    Slicing a misread region into the number of cards the caller hoped for is how a
    reader produces a full hand of confident nonsense.
    """
    region = row_of([card(i) for i in range(5)])

    assert find_row(region, expected=5) is not None
    assert find_row(region, expected=6) is None


def test_bare_felt_is_not_a_row_of_cards():
    felt = np.zeros((200, 600, 3), dtype=np.uint8)
    felt[:, :] = FELT

    assert find_row(felt) is None


# ------------------------------------------------------------------ colour ---

@pytest.mark.parametrize("colour", sorted(FRAME_REFERENCES))
def test_colour_is_read_from_the_frame(colour):
    found, distance = classify_colour(frame_colour(card(1, colour)))

    assert found == colour
    assert distance < 30


@pytest.mark.parametrize("colour", sorted(FRAME_REFERENCES))
def test_pale_artwork_spilling_over_the_frame_does_not_wash_out_the_colour(colour):
    """The measured failure: sampling one edge strip on a white-haired holomem gave
    (225,223,231), which is not any of the three frame colours."""
    found, _ = classify_colour(frame_colour(card(1, colour, overflow=True)))

    assert found == colour


def test_something_that_is_not_a_card_is_not_given_a_colour():
    grey = np.full((CARD_H, CARD_W, 3), 128, dtype=np.uint8)

    found, distance = classify_colour(frame_colour(grey))

    assert found is None
    assert distance > 0


# --------------------------------------------------------------- catalogue ---

@pytest.mark.parametrize("stem, expect", [
    ("gawr_gura_COLORLESS", "gawr_gura"),
    # The same holomem in two groups is the same picture, so the group tag is not part
    # of the identity. Collapsing them is also what keeps the margin meaningful.
    ("shirakami_fubuki_COLORLESS", "shirakami_fubuki"),
    ("shirakami_fubuki_GAMERS_COLORLESS", "shirakami_fubuki"),
    # Lowercase, punctuation and all, must survive.
    ("ninomae_ina'nis_COLORLESS", "ninomae_ina'nis"),
])
def test_group_variants_collapse_to_one_holomem(stem, expect):
    assert character_from_filename(stem) == expect


def catalogue(seeds) -> TemplateSet:
    return TemplateSet({f"holomem_{s}": [prepare(card(s))] for s in seeds})


def test_a_known_holomem_is_named():
    found = catalogue(range(8)).identify(card(3, "orange"))

    assert found.character == "holomem_3"
    assert found.confident
    assert found.margin >= MIN_MARGIN


def test_a_holomem_with_no_art_is_refused_rather_than_guessed():
    """The normal state, not an error state. The roster is redrawn every round, so a
    game can always deal somebody whose card has never been captured."""
    found = catalogue(range(8)).identify(card(999))

    assert not found.confident
    assert found.character is None
    assert found.reason


def test_two_holomem_that_look_alike_are_refused_not_picked_between():
    """Margin, not score, is the refusal signal.

    A wrong answer can score respectably -- the art is all portraits against pale
    backgrounds. What it cannot do is stand clearly apart from the field.
    """
    twin = prepare(card(3))
    ambiguous = TemplateSet({"one": [twin], "two": [twin.copy()]})

    found = ambiguous.identify(card(3))

    assert not found.confident
    assert found.margin < MIN_MARGIN


def test_an_empty_catalogue_refuses_everything():
    found = TemplateSet({}).identify(card(1))

    assert not found.confident
    assert "no card art" in found.reason


def test_the_offset_search_can_only_help():
    """The jitter grid must contain the plain crop.

    It did not: the grid was centred on the card's midpoint while the templates are cut
    from a crop centred higher, so the unjittered variant was not the plain crop and
    well-framed cards scored 0.09 *lower* with the search than without it. Silent, and
    it looks like the templates being poor.

    Asserted on the score rather than on pixels: resizing the whole card and cropping is
    equivalent in coverage to cropping and resizing, but not identical, because the
    resampling kernel sees different neighbours at the boundary. What must hold is that
    a card matched against a template cut from itself still scores ~1.
    """
    subject = card(3, "blue")
    itself = TemplateSet({"holomem_3": [prepare(subject)]})

    found = itself.identify(subject)

    assert found.score > 0.95, f"the search lost {1 - found.score:.3f} of a perfect match"
    assert len(query_variants(subject)) == 27


def test_coverage_against_a_roster_is_reported_before_a_round_starts(real_rules):
    """Because the answer decides whether the reader can advise at all, and finding
    out mid-hand means finding out as a card nobody can name."""
    from pokajan.core.roster import roster_of

    roster = roster_of(real_rules)
    known = roster.characters[:3]
    partial = TemplateSet({cid: [prepare(card(i))] for i, cid in enumerate(known)})

    missing = partial.missing_from(roster)

    assert set(missing) == set(roster.characters) - set(known)
    assert not TemplateSet({c: [prepare(card(i))] for i, c in
                            enumerate(roster.characters)}).missing_from(roster)


# ------------------------------------------------------------------ layout ---
#
# The regions themselves are checked by eye with scripts/check_layout.py, because a
# fraction is impossible to verify by reading. What is guarded here is the frame
# handling underneath them: finding the play area, and refusing when it is not there.

def letterboxed(width: int = 640, height: int = 400, bars: int = 20) -> np.ndarray:
    """A frame with black bars top and bottom, as the game is captured."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[bars:height - bars, :] = FELT
    return frame


def test_the_play_area_is_found_rather_than_assumed():
    """Every capture so far is 2880x1800 with 90-pixel bars, leaving exactly 16:9.

    Absolute pixel coordinates would be a promise about one window size, broken by a
    different monitor or a resized window.
    """
    from pokajan.vision.layout import find_play_area

    height, bars = 400, 20
    area = find_play_area(letterboxed(height=height, bars=bars))

    assert area is not None
    assert (area.x, area.y) == (0, bars)
    assert area.height == height - 2 * bars
    assert area.aspect == pytest.approx(16 / 9, abs=0.05)


def test_a_frame_that_is_not_the_game_is_refused():
    """Refusing here is far cheaper than every region afterwards being offset."""
    from pokajan.vision.layout import find_play_area

    assert find_play_area(np.zeros((400, 640, 3), dtype=np.uint8)) is None, "all black"
    assert find_play_area(np.full((400, 400, 3), 90, dtype=np.uint8)) is None, "square"
    assert find_play_area(np.zeros((4, 4, 3), dtype=np.uint8)) is None, "tiny"


def test_regions_scale_with_the_play_area():
    """The same fractions must land on the same content at any capture size."""
    from pokajan.vision.layout import HAND, find_play_area

    small = find_play_area(letterboxed(640, 400, 20))
    large = find_play_area(letterboxed(1920, 1200, 60))

    for box, area in ((HAND, small), (HAND, large)):
        left, top, right, bottom = box.pixels(area)
        assert (left / area.width) == pytest.approx(box.left, abs=0.002)
        assert ((top - area.y) / area.height) == pytest.approx(box.top, abs=0.002)
        assert right > left and bottom > top


def test_a_vertical_row_is_split_and_turned_upright():
    """The seats either side of you run their discards down the screen.

    Without this the same code returns one enormous card with an aspect around 0.42,
    which nothing downstream recognises as wrong -- it is simply a crop that never
    matches. Their real fields are also sheared, which is why the reader targets the
    newest card rather than the whole field; this covers the axis-aligned part.
    """
    from pokajan.vision.geometry import find_row

    cards = [card(i, "blue") for i in range(4)]
    sideways = np.rot90(row_of(cards), 1)          # a column, cards on their side

    row = find_row(sideways, vertical=True, rotate=-90)

    assert row is not None
    assert len(row.cards) == 4
    for one in row.cards:
        upright = one.shape[1] / one.shape[0]
        assert upright == pytest.approx(CARD_ASPECT, abs=0.05), "not turned upright"
