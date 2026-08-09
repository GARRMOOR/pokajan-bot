"""Reading the four group badges, which is what determines the roster.

Synthetic, like every vision test here: the captures are gitignored, so a fresh clone has
to pass without them. What these tests paint with is `data/captures/group_labels.yaml`,
which **is** committed -- it holds badge shapes from the game's typeface and no personal
data -- so a badge can be painted onto synthetic felt at an arbitrary size and read back.
That is a real round trip through the same code the reader uses, not a mock.

The behaviour worth guarding is refusal. A badge is the only thing that says which four
groups were dealt, and a group read wrongly is four or five characters the agent prices
for the rest of the round and never notices. Two concrete near-misses are pinned below:
`digits.DigitReader` reads "Ga" as a confident 0, and a label box that clips the ID
branches' subscript turns "1ID" into "1" -- Gen1, a real group with different members.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from pokajan.vision.group_labels import (
    LABEL_ROWS,
    LABEL_SIZE,
    MAX_INK,
    LabelError,
    LabelReader,
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


@pytest.fixture(scope="module")
def reader() -> LabelReader:
    return LabelReader.load()


@pytest.fixture(scope="module")
def book() -> GroupBook:
    return GroupBook.load()


def paint(bitmap: np.ndarray, *, width=190, height=73, scale=1.0, left=0.30, top=0.10):
    """A badge on felt: arbitrary size, arbitrary position, nothing else in the band."""
    from PIL import Image

    band = np.zeros((height, width, 3), dtype=np.uint8)
    band[:, :] = FELT

    tall = max(1, int(round(height * 0.7 * scale)))
    wide = max(1, int(round(tall * bitmap.shape[1] / bitmap.shape[0])))
    small = np.asarray(
        Image.fromarray((np.asarray(bitmap, dtype=np.float32) * 255).astype(np.uint8))
        .resize((wide, tall), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    y, x = int(top * height), int(left * width)
    wide, tall = min(wide, width - x), min(tall, height - y)
    patch = band[y:y + tall, x:x + wide]
    ink = small[:tall, :wide, None]
    patch[:] = (patch * (1 - ink) + np.asarray(INK, dtype=np.float32) * ink).astype(np.uint8)
    return band


def stack(reader: LabelReader, badges, **kwargs) -> np.ndarray:
    """Four badge bands one above the other, as the label box crops them."""
    exemplars = _exemplars()
    return np.concatenate([paint(exemplars[b], **kwargs) for b in badges], axis=0)


def _exemplars() -> dict[str, np.ndarray]:
    from pokajan.vision.group_labels import EXEMPLARS

    raw = yaml.safe_load(EXEMPLARS.read_text(encoding="utf-8"))
    return {str(k): from_text(v["rows"]) for k, v in raw["badges"].items()}


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

def test_every_committed_badge_reads_back(reader):
    """Paint each exemplar onto felt and identify it. Eight badges, eight answers."""
    assert len(reader) >= 8
    for badge, bitmap in _exemplars().items():
        found = reader.identify(paint(bitmap))
        assert found.badge == badge, f"{badge} -> {found.badge} ({found.reason})"
        assert found.score >= 0.9


@pytest.mark.parametrize("scale,left,top", [(0.5, 0.05, 0.30), (1.0, 0.30, 0.10),
                                            (1.35, 0.55, 0.02)])
def test_size_and_position_within_the_band_do_not_matter(reader, scale, left, top):
    """The ink is trimmed before matching, so where it sits is not information.

    Which is what lets the box be generous. It has to be: the badges are not all the same
    width and the ID branches hang a subscript below and to the right of the numeral.
    """
    for badge, bitmap in _exemplars().items():
        found = reader.identify(paint(bitmap, scale=scale, left=left, top=top))
        assert found.badge == badge, f"{badge} at scale {scale} -> {found.reason}"


def test_the_id_branches_are_not_their_generation_number(reader, book):
    """"1ID" is ID Gen1 and "1" is Gen1: different groups, different members, one subscript.

    This is the bug the label box was hiding. It ended at 0.560 and clipped the subscript,
    so ID Gen1 read as a confident bare 1 with nothing anywhere to object. A badge that
    loses part of itself does not refuse -- it answers, wrongly.
    """
    exemplars = _exemplars()
    assert reader.identify(paint(exemplars["1"])).badge == "1"
    assert reader.identify(paint(exemplars["1ID"])).badge == "1ID"

    assert book.by_badge("1").id == "gen1"
    assert book.by_badge("1ID").id == "id1"
    assert set(book.by_badge("1").members).isdisjoint(book.by_badge("1ID").members)


def test_a_digit_reader_is_not_a_badge_reader():
    """Measured, and the reason this module exists rather than reusing digits.py.

    A digit alphabet has nothing for a G to compete against, so "Ga" scores 0.74 as a 0
    with its runner-up 0.25 behind -- comfortably past every threshold digits.py has. The
    refusal machinery cannot engage when the right answer is not in the alphabet.
    """
    from pokajan.vision.digits import DigitReader

    digits = DigitReader.load()
    if not len(digits):
        pytest.skip("no digit exemplars")
    assert "Ga" not in digits._digits
    assert set(digits._digits) == set("0123456789")


# ----------------------------------------------------------------- refusals ----

def test_an_unseen_badge_is_refused(reader):
    """Seven of the fifteen badges have never been captured, and they must not be guessed.

    A cross painted on felt is not any of them, and nothing in the alphabet may claim it.
    """
    cross = np.zeros((28, 28), dtype=np.float32)
    cross[12:16, :] = 1.0
    cross[:, 12:16] = 1.0

    found = reader.identify(paint(cross))

    assert not found.confident, f"claimed {found.badge} at {found.score:.2f}"


def test_a_blank_band_is_refused(reader):
    band = np.zeros((73, 190, 3), dtype=np.uint8)
    band[:, :] = FELT

    assert reader.identify(band).reason == "nothing written here"


def test_a_band_buried_under_a_payout_panel_is_refused(reader):
    """The payout panels are large and near-white and sit right over these rows.

    Measured on real frames: a covered band comes back 47% to 84% ink, where a badge is a
    few per cent. Refusing on ink alone means this never reaches the matcher at all.
    """
    band = np.zeros((73, 190, 3), dtype=np.uint8)
    band[:, :] = FELT
    band[:, 20:] = (245, 245, 248)

    found = reader.identify(band)

    assert not found.confident
    assert "covering it" in found.reason
    assert float(ink_mask(band).mean()) > MAX_INK


def test_a_reader_with_no_exemplars_refuses_rather_than_matching_nothing():
    found = LabelReader({}).identify(np.zeros((73, 190, 3), dtype=np.uint8))

    assert not found.confident
    assert "no badge exemplars" in found.reason


def test_a_crop_too_short_for_four_rows_is_refused():
    with pytest.raises(LabelError):
        split_rows(np.zeros((2, 40, 3), dtype=np.uint8))


# ------------------------------------------------------------------- shapes ----

def test_split_rows_gives_four_equal_bands():
    """Equal bands, not found ones. There are always four rows and a covered panel has
    no structure to find -- so looking for one would fail exactly when it matters."""
    bands = split_rows(np.zeros((80, 40, 3), dtype=np.uint8))

    assert len(bands) == LABEL_ROWS
    assert [b.shape[0] for b in bands] == [20, 20, 20, 20]


def test_aspect_is_carried_into_the_canvas_not_normalised_away():
    """A narrow badge stays narrow. Stretching each to fill the canvas would make "1" and
    "My" the same shape, and the alphabet is too small to give away a whole dimension."""
    narrow = np.zeros((40, 10), dtype=np.float32)
    narrow[:, :] = 1.0
    wide = np.zeros((10, 40), dtype=np.float32)
    wide[:, :] = 1.0

    tall_canvas, flat_canvas = normalise(narrow), normalise(wide)

    assert tall_canvas.shape == (LABEL_SIZE[1], LABEL_SIZE[0])
    assert not tall_canvas[:, 0].any(), "a narrow badge should be letterboxed sideways"
    assert not flat_canvas[0, :].any(), "a wide badge should be letterboxed vertically"
    assert tall_canvas.sum() == pytest.approx(flat_canvas.sum(), rel=0.2)


def test_the_ink_range_comes_from_the_whole_band(reader):
    """The bug `digits.ink_mask` documents, guarded here too.

    Take the brightness range over the pale pixels alone and a band whose only pale thing
    is the badge has no range at all: the contrast test calls it blank and the badge
    disappears. Painted here in a flat uniform grey with nothing else unsaturated, which is
    the case real captures hide behind antialiasing.
    """
    band = np.zeros((73, 190, 3), dtype=np.uint8)
    band[:, :] = (20, 130, 20)
    band[20:55, 60:120] = (210, 212, 208)

    mask = ink_mask(band)

    assert 0.05 < mask.mean() < 0.25
    assert mask[35, 90] and not mask[5, 5]


# --------------------------------------------------------------- end to end ----

def test_panel_and_badges_together_give_a_roster(reader, book):
    """Four badges in, a roster out, no portrait recognised anywhere."""
    roster = read_roster(
        synthetic_panel([4, 4, 4, 5]), stack(reader, ["Ga", "4", "5", "My"]),
        book=book, reader=reader, bonus="gawr_gura",
    )

    assert [g.id for g in roster.groups] == ["gamers", "gen4", "gen5", "myth"]
    assert len(roster.characters) == 17


def test_a_badge_that_disagrees_with_its_row_is_refused(reader, book):
    """Myth has five members, so reading it above a four-cell row is a misread.

    The count is measured without recognising anything, which is exactly why it can
    contradict the badge. Nothing else in the read can.
    """
    with pytest.raises(PanelError, match="misread"):
        read_roster(synthetic_panel([4, 4, 4, 4]), stack(reader, ["Ga", "4", "5", "My"]),
                    book=book, reader=reader)


def test_a_covered_panel_refuses_before_any_badge_is_read(reader, book):
    """[4, 3, 2, 2] is what a real mid-payout frame counted. 2 is not a group size."""
    with pytest.raises(PanelError, match="covered"):
        read_roster(synthetic_panel([4, 3, 2, 2]), stack(reader, ["Ga", "4", "5", "My"]),
                    book=book, reader=reader)


def test_one_unreadable_badge_refuses_the_whole_roster(reader, book):
    """A partial roster is not a roster: the missing group's characters are in the deck
    either way, and an agent that does not know about them prices every hand wrongly."""
    labels = stack(reader, ["Ga", "4", "5", "My"])
    labels[3 * 73:] = (245, 245, 248)          # a payout panel over the last row

    with pytest.raises(PanelError, match="could not read every badge"):
        read_roster(synthetic_panel([4, 4, 4, 5]), labels, book=book, reader=reader)


# ------------------------------------------------------------------- table ----

def test_every_committed_badge_belongs_to_a_group(reader, book):
    """An exemplar for a badge no group prints would be unmappable at read time."""
    for badge in reader.badges:
        assert book.by_badge(badge).badge == badge


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


def test_the_badges_still_missing_are_named_not_counted(reader, book):
    """Closing the gap means capturing a frame that shows one, which is only actionable
    if the player is told which. Seven of fifteen are still unseen."""
    missing = reader.uncovered(book)

    assert set(missing) <= {group.badge for group in book.groups}
    assert set(missing).isdisjoint(reader.badges)
