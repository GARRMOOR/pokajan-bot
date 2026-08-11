"""Reading the four group badges, which is what determines the roster.

Synthetic, like every vision test here: the captures are gitignored, so a fresh clone has
to pass without them. What these tests paint with is `data/captures/group_labels.yaml`,
which **is** committed -- it holds glyph shapes from the game's typeface and no personal
data -- so a badge can be assembled onto synthetic felt at an arbitrary size and read back.
That is a real round trip through the same code the reader uses, not a mock.

The behaviour worth guarding is refusal. A badge is the only thing that says which four
groups were dealt, and a group read wrongly is four or five characters the agent prices for
the rest of the round and never notices. Three concrete near-misses are pinned below:

* `digits.DigitReader` reads "Ga" as a confident 0;
* a label box that clips the ID branches' subscript turns "1ID" into "1" -- Gen1, a real
  group with different members;
* matching each badge as one picture left "2ID" and "3ID" 0.14 apart, and since ID Gen2 and
  ID Gen3 both have three members the panel's member count cannot break that tie either.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from pokajan.vision.group_labels import (
    LABEL_ROWS,
    LABEL_SIZE,
    MAX_INK,
    SUBSCRIPT_NAMES,
    LabelError,
    LabelReader,
    _split_badge,
    decompose,
    from_text,
    ink_mask,
    normalise,
    split_rows,
)
from pokajan.vision.roster_panel import (
    PANEL_COLUMNS,
    PANEL_ROWS,
    GroupBook,
    PanelError,
    read_roster,
)

pytestmark = pytest.mark.invariant

FELT = (34, 110, 34)
INK = (228, 231, 225)
PLACEHOLDER = (214, 214, 219)
BAND = (73, 190)


@pytest.fixture(scope="module")
def reader() -> LabelReader:
    return LabelReader.load()


@pytest.fixture(scope="module")
def book() -> GroupBook:
    return GroupBook.load()


@pytest.fixture(scope="module")
def shapes() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from pokajan.vision.group_labels import EXEMPLARS

    raw = yaml.safe_load(EXEMPLARS.read_text(encoding="utf-8"))
    read = lambda key: {str(k): from_text(v["rows"])            # noqa: E731
                        for k, v in (raw.get(key) or {}).items()}
    return read("primaries"), read("subscripts")


def stamp(band, bitmap, *, left, top, height):
    """Paint a glyph onto a band in ink, scaled to `height` fraction of the band."""
    from PIL import Image

    rows, cols = band.shape[0], band.shape[1]
    tall = max(2, int(round(rows * height)))
    wide = max(2, int(round(tall * bitmap.shape[1] / bitmap.shape[0])))
    small = np.asarray(
        Image.fromarray((np.asarray(bitmap, dtype=np.float32) * 255).astype(np.uint8))
        .resize((wide, tall), Image.BILINEAR), dtype=np.float32) / 255.0

    y, x = int(top * rows), int(left * cols)
    tall, wide = min(tall, rows - y), min(wide, cols - x)
    patch = band[y:y + tall, x:x + wide]
    alpha = small[:tall, :wide, None]
    patch[:] = (patch * (1 - alpha) + np.asarray(INK, dtype=np.float32) * alpha).astype(
        np.uint8)
    return x + wide


def compose(shapes, badge, *, scale=1.0, left=0.12, size=BAND):
    """A whole badge on felt: primary, then its subscript low and to the right."""
    primaries, subscripts = shapes
    primary_name, subscript_name = _split_badge(badge)
    band = np.zeros((*size, 3), dtype=np.uint8)
    band[:, :] = FELT

    end = stamp(band, primaries[primary_name],
                left=left, top=0.08, height=0.72 * scale)
    if subscript_name:
        stamp(band, subscripts[subscript_name],
              left=end / size[1] + 0.03, top=0.08 + 0.40 * scale, height=0.34 * scale)
    return band


def stack(shapes, badges, **kwargs) -> np.ndarray:
    """Four badge bands one above the other, as the label box crops them."""
    return np.concatenate([compose(shapes, b, **kwargs) for b in badges], axis=0)


def synthetic_panel(sizes, *, width=400, height=360) -> np.ndarray:
    """The 4x5 grid: detailed art in filled cells, flat grey placeholders in the rest."""
    made = np.zeros((height, width, 3), dtype=np.uint8)
    made[:, :] = FELT
    rng = np.random.default_rng(4)
    for row, filled in enumerate(sizes):
        top, bottom = height * row // PANEL_ROWS, height * (row + 1) // PANEL_ROWS
        for column in range(PANEL_COLUMNS):
            left = width * column // PANEL_COLUMNS
            right = width * (column + 1) // PANEL_COLUMNS
            cell = made[top + 4:bottom - 4, left + 4:right - 4]
            cell[:] = (rng.integers(0, 255, size=(*cell.shape[:2], 3))
                       if column < filled else PLACEHOLDER)
    return made


# ------------------------------------------------------------------ reading ----

def test_every_badge_the_game_prints_reads_back(reader, book, shapes):
    """All fifteen, assembled onto felt and identified. None may refuse and none may lie."""
    assert len(reader.badges) == len(book.groups) == 15
    for group in book.groups:
        found = reader.identify(compose(shapes, group.badge))
        assert found.badge == group.badge, f"{group.badge} -> {found.badge} ({found.reason})"
        assert found.score >= 0.85


@pytest.mark.parametrize("scale,left", [(0.6, 0.03), (1.0, 0.12), (1.25, 0.30)])
def test_size_and_position_within_the_band_do_not_matter(reader, book, shapes, scale, left):
    """Each glyph is trimmed before matching, so where it sits carries no information.

    Which is what lets the box be generous, and it has to be: the badges differ in width and
    the ID branches hang a subscript below and to the right of the numeral. Real captures
    span a fourfold scale change between the table and the reveal screen.
    """
    for group in book.groups:
        found = reader.identify(compose(shapes, group.badge, scale=scale, left=left))
        assert found.badge == group.badge, \
            f"{group.badge} at scale {scale} -> {found.badge} ({found.reason})"


def test_the_id_branches_are_not_their_generation_number(reader, book, shapes):
    """"1ID" is ID Gen1 and "1" is Gen1: different groups, different members, one subscript.

    This is the bug the label box was hiding. It ended at 0.560 and clipped the subscript, so
    ID Gen1 read as a confident bare 1 with nothing anywhere to object. A badge that loses
    part of itself does not refuse -- it answers, wrongly.
    """
    for numeral in ("1", "2", "3"):
        bare = reader.identify(compose(shapes, numeral))
        suffixed = reader.identify(compose(shapes, numeral + "ID"))
        assert bare.badge == numeral and not bare.subscript
        assert suffixed.badge == numeral + "ID" and suffixed.subscript == "ID"

    assert book.by_badge("1").id == "gen1"
    assert book.by_badge("1ID").id == "id1"
    assert set(book.by_badge("1").members).isdisjoint(book.by_badge("1ID").members)


def test_the_id_generations_are_told_apart_by_their_numeral(reader, book, shapes):
    """ID Gen2 and ID Gen3 both have three members, so the member count cannot separate them.

    This is the pair that forced the primary and the subscript to be matched separately.
    Whole-badge matching put them 0.86 apart -- 0.14 of margin, under the threshold -- because
    the identical "ID" is most of the picture. Raising the canvas from 28px to 96px did not
    help, since resolution does not change a ratio of shared to distinguishing ink.
    """
    two, three = (reader.identify(compose(shapes, b)) for b in ("2ID", "3ID"))

    assert (two.badge, three.badge) == ("2ID", "3ID")
    assert min(two.margin, three.margin) >= 0.15
    assert book.by_id("id2").size == book.by_id("id3").size == 3
    assert set(book.by_id("id2").members).isdisjoint(book.by_id("id3").members)


def test_a_digit_reader_is_not_a_badge_reader():
    """Measured, and the reason this module exists rather than reusing digits.py.

    A digit alphabet has nothing for a G to compete against, so "Ga" scores 0.74 as a 0 with
    its runner-up 0.25 behind -- comfortably past every threshold digits.py has. Refusal
    cannot engage when the right answer is not in the alphabet.
    """
    from pokajan.vision.digits import DigitReader

    digits = DigitReader.load()
    if not len(digits):
        pytest.skip("no digit exemplars")
    assert set(digits._digits) == set("0123456789")


# -------------------------------------------------------------- decomposing ----

def test_a_bare_numeral_has_no_subscript(shapes):
    parts = decompose(ink_mask(compose(shapes, "4")))

    assert parts is not None and parts.subscript is None


def test_a_subscript_is_found_low_and_to_the_right(shapes):
    """By geometry, not recognition -- which is what lets it be matched independently.

    Measured across both screens and a fourfold scale change: an "ID" begins at 0.48-0.51 of
    the badge's height while every primary begins at 0.00, so the 0.30 line has a wide berth.
    """
    parts = decompose(ink_mask(compose(shapes, "3ID")))

    assert parts is not None and parts.subscript is not None
    assert parts.subscript.shape[1] < parts.primary.shape[1]


def test_a_speck_below_the_numeral_is_not_a_subscript(reader, shapes):
    """Otherwise a stray antialiased pixel would turn Gen1 into ID Gen1."""
    band = compose(shapes, "1")
    band[60:63, 150:154] = INK              # low and to the right, but tiny

    parts = decompose(ink_mask(band))

    assert parts is not None and parts.subscript is None
    assert reader.identify(band).badge == "1"


def test_the_letter_badges_come_through_as_one_glyph(shapes):
    """Their second letter touches the first, so there is nothing to split.

    Hence twelve primaries rather than a primary alphabet of single letters -- "Ga" is its
    own shape. If one ever did split, its primary would be a bare "G", which has no exemplar,
    and the read would refuse rather than guess.
    """
    for badge in ("Ga", "My", "Ad", "Pr", "Re"):
        assert _split_badge(badge) == (badge, "")


def test_a_subscript_that_matches_nothing_is_refused(reader, shapes):
    """Present but unrecognised must refuse, not be ignored.

    Ignoring it would read ID Gen2 as Gen2. Only "ID" separates from its primary today, so
    detecting mere presence would be enough -- but matching it costs nothing and does not
    assume the game will never print another.
    """
    band = compose(shapes, "2")
    band[38:66, 120:170] = INK              # a solid block where an "ID" would sit

    found = reader.identify(band)

    assert not found.confident, f"claimed {found.badge}"
    assert "subscript" in found.reason


def test_a_primary_and_subscript_that_compose_to_nothing_are_refused(reader, shapes):
    """"Ga" with an "ID" beneath it is not a badge, and "GaID" is worse than a refusal."""
    primaries, subscripts = shapes
    band = compose(shapes, "Ga")
    end = int(0.62 * BAND[1])
    stamp(band, subscripts["ID"], left=end / BAND[1], top=0.50, height=0.34)

    found = reader.identify(band)

    assert not found.confident
    assert "no group prints" in found.reason


# ----------------------------------------------------------------- refusals ----

def test_an_unknown_glyph_is_refused(reader):
    """A cross is not any of the twelve primaries, and nothing may claim it."""
    band = np.zeros((*BAND, 3), dtype=np.uint8)
    band[:, :] = FELT
    band[20:52, 70:120] = FELT
    for offset in range(32):
        band[20 + offset, 70 + offset * 3 // 2] = INK
        band[20 + offset, 118 - offset * 3 // 2] = INK

    found = reader.identify(band)

    assert not found.confident, f"claimed {found.badge} at {found.score:.2f}"


def test_a_blank_band_is_refused(reader):
    band = np.zeros((*BAND, 3), dtype=np.uint8)
    band[:, :] = FELT

    assert reader.identify(band).reason == "nothing written here"


def test_a_band_buried_under_a_payout_panel_is_refused(reader):
    """The payout panels are large and near-white and sit right over these rows.

    Measured on real frames: a covered band comes back 47% to 84% ink where a badge is a few
    per cent, so refusing on ink alone means this never reaches the matcher at all.
    """
    band = np.zeros((*BAND, 3), dtype=np.uint8)
    band[:, :] = FELT
    band[:, 20:] = (245, 245, 248)

    found = reader.identify(band)

    assert not found.confident
    assert "covering it" in found.reason
    assert float(ink_mask(band).mean()) > MAX_INK


def test_a_reader_with_no_exemplars_refuses_rather_than_matching_nothing():
    found = LabelReader({}).identify(np.zeros((*BAND, 3), dtype=np.uint8))

    assert not found.confident
    assert "no badge exemplars" in found.reason


def test_a_crop_too_short_for_four_rows_is_refused():
    with pytest.raises(LabelError):
        split_rows(np.zeros((2, 40, 3), dtype=np.uint8))


# ------------------------------------------------------------------- shapes ----

def test_split_rows_gives_four_equal_bands():
    """Equal bands, not found ones. There are always four rows, and a covered panel has no
    structure to find -- so looking for one would fail exactly when it matters."""
    bands = split_rows(np.zeros((80, 40, 3), dtype=np.uint8))

    assert len(bands) == LABEL_ROWS
    assert [b.shape[0] for b in bands] == [20, 20, 20, 20]


def test_a_glyph_is_stretched_to_fill_the_canvas():
    """Not letterboxed, and this is the measured direction rather than the intuitive one.

    Preserving aspect and padding the shorter side was tried first and is worse at every
    canvas size from 12x18 to 36x36: mean pairwise score 0.31 against 0.20, and the tightest
    margin on a real sighting 0.15 against 0.27. Identical padding *correlates*, so two
    badges sharing nothing but their empty margins still agree over those margins.
    """
    narrow = np.ones((40, 6), dtype=np.float32)
    wide = np.ones((6, 40), dtype=np.float32)

    for canvas in (normalise(narrow), normalise(wide)):
        assert canvas.shape == (LABEL_SIZE[1], LABEL_SIZE[0])
        assert canvas[0, :].all() and canvas[-1, :].all()
        assert canvas[:, 0].all() and canvas[:, -1].all()


def test_the_ink_range_comes_from_the_whole_band():
    """The bug `digits.ink_mask` documents, guarded here too.

    Take the brightness range over the pale pixels alone and a band whose only pale thing is
    the badge has no range at all: the contrast test calls it blank and the badge disappears.
    Painted here in a flat uniform grey with nothing else unsaturated, which is the case real
    captures hide behind antialiasing.
    """
    band = np.zeros((*BAND, 3), dtype=np.uint8)
    band[:, :] = (20, 130, 20)
    band[20:55, 60:120] = (210, 212, 208)

    mask = ink_mask(band)

    assert 0.05 < mask.mean() < 0.25
    assert mask[35, 90] and not mask[5, 5]


# --------------------------------------------------------------- end to end ----

def test_panel_and_badges_together_give_a_roster(reader, book, shapes):
    """Four badges in, a roster out, no portrait recognised anywhere."""
    roster = read_roster(
        synthetic_panel([4, 4, 4, 5]), stack(shapes, ["Ga", "4", "5", "My"]),
        book=book, reader=reader, bonus="gawr_gura",
    )

    assert [g.id for g in roster.groups] == ["gamers", "gen4", "gen5", "myth"]
    assert len(roster.characters) == 17


def test_a_badge_that_disagrees_with_its_row_is_refused(reader, book, shapes):
    """Myth has five members, so reading it above a four-cell row is a misread.

    The count is measured without recognising anything, which is exactly why it can
    contradict the badge. Nothing else in the read can.
    """
    with pytest.raises(PanelError, match="misread"):
        read_roster(synthetic_panel([4, 4, 4, 4]), stack(shapes, ["Ga", "4", "5", "My"]),
                    book=book, reader=reader)


def test_a_covered_panel_refuses_before_any_badge_is_read(reader, book, shapes):
    """[4, 3, 2, 2] is what a real mid-payout frame counted. 2 is not a group size."""
    with pytest.raises(PanelError, match="covered"):
        read_roster(synthetic_panel([4, 3, 2, 2]), stack(shapes, ["Ga", "4", "5", "My"]),
                    book=book, reader=reader)


def test_one_unreadable_badge_refuses_the_whole_roster(reader, book, shapes):
    """A partial roster is not a roster: the missing group's characters are in the deck
    either way, and an agent that does not know about them prices every hand wrongly."""
    labels = stack(shapes, ["Ga", "4", "5", "My"])
    labels[3 * BAND[0]:] = (245, 245, 248)         # a payout panel over the last row

    with pytest.raises(PanelError, match="could not read every badge"):
        read_roster(synthetic_panel([4, 4, 4, 5]), labels, book=book, reader=reader)


# ------------------------------------------------------------------- table ----

def test_the_reader_composes_every_badge_the_game_prints(reader, book):
    """Fifteen of fifteen, and `uncovered` is how a gap becomes a capture request."""
    assert reader.uncovered(book) == []
    assert set(reader.badges) == {group.badge for group in book.groups}


def test_a_reader_missing_a_primary_reports_the_badges_it_cannot_compose(reader, book,
                                                                        shapes):
    """Dropping "3" must cost Gen3 *and* ID Gen3, since both are built from it."""
    primaries, subscripts = shapes
    thinner = LabelReader({k: v for k, v in primaries.items() if k != "3"},
                          subscripts, [g.badge for g in book.groups])

    assert thinner.uncovered(book) == ["3", "3ID"]


def test_badges_identify_a_group_on_their_own(book, tmp_path):
    """Two groups sharing a badge would make the lookup a coin toss, so loading refuses.

    Nearly shipped: all three ID branches were recorded with badge "ID" before a capture
    showed the game prints the generation number with "ID" beneath it.
    """
    badges = [group.badge for group in book.groups]
    assert len(badges) == len(set(badges))

    clash = tmp_path / "clash.yaml"
    clash.write_text(yaml.safe_dump({"groups": [
        {"id": "a", "label": "A", "badge": "ID", "size": 3, "members": ["x", "y", "z"]},
        {"id": "b", "label": "B", "badge": "ID", "size": 3, "members": ["p", "q", "r"]},
    ]}), encoding="utf-8")

    with pytest.raises(PanelError, match="more than one group"):
        GroupBook.load(clash)


def test_a_primary_plus_subscript_names_exactly_one_group(book):
    """The composition rule only works because (primary, has subscript) is unique.

    Every letter primary is unique on its own, and each numeral appears at most twice --
    once bare as a Gen and once with "ID". If the game ever printed a second subscript on a
    numeral this assumption would need revisiting, which is why it is asserted rather than
    assumed.
    """
    seen = set()
    for group in book.groups:
        primary, subscript = _split_badge(group.badge)
        assert (primary, subscript) not in seen, group.badge
        seen.add((primary, subscript))
        assert subscript in ("", *SUBSCRIPT_NAMES)
