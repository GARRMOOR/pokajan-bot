"""Reading a roster out of the group panel, without recognising any portraits.

Matching the panel's head-and-shoulders portraits against the card art scores 1/17 -- a
different rendering, not a crop -- so the roster comes from the four group *labels*
instead, looked up in `data/captures/hololive_groups.yaml`. That table is committed and is
public information about an agency, so unlike the captures it can be tested directly.

What the tests guard is the check that makes the lookup trustworthy: the panel's member
counts are measured without recognising anything, so they are independent evidence about
the labels. When they disagree, the only safe answer is to refuse -- a wrong roster loads
happily and then misprices the whole game.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokajan.core.roster import rules_for_roster
from pokajan.vision.roster_panel import (
    PANEL_COLUMNS,
    PANEL_ROWS,
    GroupBook,
    PanelError,
    count_members,
    panel_is_readable,
    roster_from_groups,
)

pytestmark = pytest.mark.invariant

FELT = (34, 139, 34)
PLACEHOLDER = (214, 214, 219)


@pytest.fixture(scope="module")
def book() -> GroupBook:
    return GroupBook.load()


def panel(sizes, *, width=400, height=360) -> np.ndarray:
    """A synthetic 4x5 grid: detailed colour art in filled cells, flat grey placeholders."""
    made = np.zeros((height, width, 3), dtype=np.uint8)
    made[:, :] = FELT
    rng = np.random.default_rng(4)
    for row, filled in enumerate(sizes):
        top, bottom = height * row // PANEL_ROWS, height * (row + 1) // PANEL_ROWS
        for column in range(PANEL_COLUMNS):
            left = width * column // PANEL_COLUMNS
            right = width * (column + 1) // PANEL_COLUMNS
            cell = made[top + 4:bottom - 4, left + 4:right - 4]
            if column < filled:
                cell[:] = rng.integers(0, 255, size=(*cell.shape[:2], 3))
            else:
                cell[:] = PLACEHOLDER
    return made


# ------------------------------------------------------------------- table ---

def test_the_group_table_agrees_with_itself(book):
    """Each group's declared size must match its member list.

    Loading raises otherwise, because every size check downstream would then be
    validating against a lie.
    """
    for group in book.groups:
        assert len(group.members) == group.size, group.id
        assert group.label and group.badge


def test_every_group_size_is_one_the_game_can_show(book):
    """3, 4 or 5. Every roster shape observed in a real round fits those."""
    assert {group.size for group in book.groups} <= {3, 4, 5}


@pytest.mark.parametrize("shape", [(4, 4, 4, 5), (4, 4, 3, 5), (4, 4, 4, 3)])
def test_observed_roster_shapes_are_all_buildable(book, shape):
    """The shapes seen in real games must each be reachable from the table."""
    by_size: dict[int, list] = {}
    for group in book.groups:
        by_size.setdefault(group.size, []).append(group.id)
    for size in shape:
        assert by_size.get(size), f"no group of size {size}"


def test_a_roster_read_from_labels_builds_a_playable_game(book, real_rules):
    """The whole point: four labels in, a `Rules` out, no portrait recognised."""
    roster = roster_from_groups(
        book, ["gamers", "gen4", "gen5", "myth"],
        member_counts=[4, 4, 4, 5], bonus="gawr_gura",
    )
    rules = rules_for_roster(real_rules, roster)

    assert rules.cards.n_chars == 17
    assert [len(m) for m in rules.cards.group_members] == [4, 4, 4, 5]
    assert rules.cards.character_ids[rules.bonus_character] == "gawr_gura"


# ---------------------------------------------------------------- counting ---

@pytest.mark.parametrize("sizes", [(4, 4, 4, 5), (4, 4, 3, 5), (5, 5, 5, 5), (3, 3, 3, 3)])
def test_members_are_counted_without_recognising_anything(sizes):
    """Placeholders are flat and grey; artwork is neither. That is the whole test.

    It matters that this needs no recognition: it is the independent evidence a group
    label is checked against, so it must not depend on the thing it checks.
    """
    assert count_members(panel(list(sizes))) == list(sizes)


def test_a_count_no_group_could_have_means_the_panel_is_unreadable(book):
    """Which in practice means a payout is being displayed over it.

    Measured: a frame mid-payout counted [4, 3, 2, 2] where the clean frames either side
    both counted [4, 4, 4, 5]. The reader has to know when it cannot see.
    """
    assert panel_is_readable([4, 4, 4, 5], book)
    assert panel_is_readable([4, 4, 3, 5], book)
    assert not panel_is_readable([4, 3, 2, 2], book)
    assert not panel_is_readable([4, 4, 4], book)


def test_a_panel_crop_too_small_to_hold_a_grid_is_refused():
    with pytest.raises(PanelError):
        count_members(np.zeros((2, 2, 3), dtype=np.uint8))


# --------------------------------------------------------------- refusals ----

def test_a_label_that_disagrees_with_the_panel_is_refused(book):
    """Gen3 has four members, so reading it against a five-cell row is a misread.

    Refusing is the only safe response: the roster would otherwise load happily and
    misprice every hand for the rest of the round.
    """
    with pytest.raises(PanelError, match="misread"):
        roster_from_groups(book, ["gamers", "gen4", "gen5", "gen3"],
                           member_counts=[4, 4, 4, 5])


def test_a_group_of_the_right_size_is_accepted(book):
    """The check is on size, so any same-sized group passes it.

    Worth being explicit that this is a *cross-check* and not identification -- it
    catches a label read as a differently-sized group, which is most misreads, and
    cannot catch one read as a group of the same size.
    """
    roster = roster_from_groups(book, ["gamers", "gen4", "gen5", "gen0"],
                               member_counts=[4, 4, 4, 5])

    assert len(roster.characters) == 17


def test_mismatched_label_and_row_counts_are_refused(book):
    with pytest.raises(PanelError, match="rows of members"):
        roster_from_groups(book, ["gamers", "gen4"], member_counts=[4, 4, 4, 5])


def test_an_unknown_group_is_refused(book):
    with pytest.raises(PanelError, match="unknown group"):
        roster_from_groups(book, ["gamers", "gen4", "gen5", "holoEN9"])


# ------------------------------------------------------------------ naming ----

def test_card_art_filenames_resolve_to_canonical_ids(book):
    """The art is named by hand, and the names drift.

    Punctuation is folded away rather than aliased one name at a time, so a filename
    keeping an apostrophe resolves on its own -- as will the next name like it.
    """
    assert book.resolve("gawr_gura") == "gawr_gura"
    assert book.resolve("GAWR_GURA") == "gawr_gura"
    assert book.resolve("ninomae_ina'nis") == "ninomae_inanis"
    assert book.resolve("la+plus_darkness") == book.resolve("laplus_darkness")


def test_the_roster_covers_every_group_the_game_draws(book):
    """Including the two that only turned up when the player listed what art they lacked.

    An unknown group refuses rather than guessing, so a missing branch is safe but
    useless -- the reader simply cannot read a round that deals it.
    """
    labels = {group.id for group in book.groups}

    assert {"advent", "regloss"} <= labels, "EN Advent and DEV_IS ReGLOSS are dealt"
    assert len(book.characters) == 62


def test_graduated_members_are_left_out(book):
    """The game deals the current roster, so Gen1 and Gen2 are fours, not fives.

    Inferred from the art the player holds against the list of what they still need:
    every holomem the game can deal is one they expect to capture, and Mel and Aqua are
    on neither list. Consistent with Gen3 without Rushia and Gen4 without Coco, which the
    observed panel counts had already pinned at four.
    """
    for group_id, size in (("gen1", 4), ("gen2", 4), ("gen3", 4), ("gen4", 4)):
        assert book.by_id(group_id).size == size
    assert "yozora_mel" not in book.characters
    assert "minato_aqua" not in book.characters


def test_an_unrecognised_filename_is_reported_not_guessed(book):
    """`usada_pekore` is one letter from a real holomem, and that is exactly why.

    A resolver willing to close a one-letter gap would as happily map a genuine holomem
    onto the wrong one, and a catalogue keyed on a wrong id is a holomem the reader can
    never name. Unrecognised is useful; wrong is not.
    """
    assert book.resolve("shirakami_fubuko") is None

    resolved, unresolved, missing = book.coverage(
        ["gawr_gura", "ninomae_ina'nis", "not_a_holomem"]
    )

    assert resolved == {"gawr_gura": "gawr_gura", "ninomae_ina'nis": "ninomae_inanis"}
    assert unresolved == ["not_a_holomem"]
    assert "mori_calliope" in missing
