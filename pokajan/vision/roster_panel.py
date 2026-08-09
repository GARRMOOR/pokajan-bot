"""Turning the group panel into a roster, without recognising a single portrait.

The panel shows four groups and their members. Matching those head-and-shoulders
portraits against the card art scores 1/17 -- they are a different rendering, not a crop
-- so reading them directly would need a whole second template set.

It does not have to. The game draws real hololive branches, so **which four groups were
drawn is enough to know the roster**, and membership lives in the committed
`data/captures/hololive_groups.yaml`. That reduces up to twenty portrait matches to a
four-way label classification and a lookup.

The lookup is only trustworthy because it is checked. The panel is a fixed 4x5 grid whose
unused cells hold a flat grey placeholder, so counting the real cells in a row needs no
recognition at all -- and a count that disagrees with the group's known size means the
label was misread. That disagreement must refuse rather than repair: a wrong roster loads
happily and then misprices the entire game, and advice from it looks exactly like advice
from a right one.

This module also resolves the card-art filenames, which are typed by hand as captures come
in and do not always match the canonical ids. It maps what it can and **reports what it
cannot**, because a catalogue that silently keys on a typo is a holomem the reader can
never name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from typing import TYPE_CHECKING

from ..core.roster import ObservedGroup, ObservedRoster

if TYPE_CHECKING:
    from .group_labels import LabelReader

GROUPS_FILE = Path(__file__).resolve().parents[2] / "data" / "captures" / "hololive_groups.yaml"

# A placeholder cell is flat and grey: near-zero saturation, mid brightness, and almost no
# variation. Artwork is none of those.
PLACEHOLDER_MAX_SATURATION = 0.18
PLACEHOLDER_MAX_DETAIL = 26.0        # standard deviation of brightness within the cell
PANEL_ROWS, PANEL_COLUMNS = 4, 5


class PanelError(ValueError):
    """The panel could not be read into a roster."""


def _normalise_id(name: str) -> str:
    """Fold away the punctuation that hand-typed filenames disagree about.

    Apostrophes and plus signs appear in real holomem names, and whether a filename keeps
    them is a coin toss. Everything else is left alone -- this normalises spelling, it does
    not guess at it.
    """
    return "".join(c for c in name.strip().lower() if c.isalnum() or c == "_")


@dataclass(frozen=True)
class Group:
    """One hololive branch the game can draw."""

    id: str
    label: str
    badge: str
    size: int
    members: tuple[str, ...]
    confirmed: bool = False


@dataclass(frozen=True)
class GroupBook:
    """Every group the game can draw, and how to name a holomem."""

    groups: tuple[Group, ...]
    aliases: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = GROUPS_FILE) -> "GroupBook":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        groups = []
        for entry in raw.get("groups") or []:
            members = tuple(entry["members"])
            if len(members) != int(entry["size"]):
                # The file contradicts itself, which would make every size check
                # meaningless. Better to fail at load than to check against a lie.
                raise PanelError(
                    f"group {entry['id']!r} lists {len(members)} members "
                    f"but claims size {entry['size']}"
                )
            groups.append(Group(
                id=str(entry["id"]), label=str(entry["label"]), badge=str(entry["badge"]),
                size=int(entry["size"]), members=members,
                confirmed=entry.get("confirmed") == "capture",
            ))

        # Badges have to be unique, because the badge is what a read produces and the
        # group is what it has to mean. Two groups sharing one would make that lookup a
        # coin toss -- and it nearly happened: the ID branches were recorded as badge "ID"
        # for all three before a capture showed the game actually prints "1ID", "2ID" and
        # "3ID". A silent duplicate there would have picked whichever came first.
        badges = [group.badge for group in groups]
        duplicated = sorted({b for b in badges if badges.count(b) > 1})
        if duplicated:
            raise PanelError(
                f"badge(s) {', '.join(duplicated)} are used by more than one group -- "
                f"a badge has to identify a group on its own"
            )
        return cls(groups=tuple(groups), aliases=dict(raw.get("aliases") or {}))

    def by_id(self, group_id: str) -> Group:
        for group in self.groups:
            if group.id == group_id:
                return group
        raise PanelError(f"unknown group {group_id!r}")

    def by_badge(self, badge: str) -> Group:
        for group in self.groups:
            if group.badge == badge:
                return group
        raise PanelError(f"no group prints the badge {badge!r}")

    @property
    def characters(self) -> frozenset[str]:
        """Every canonical holomem id, which is what card art has to resolve to."""
        return frozenset(m for group in self.groups for m in group.members)

    # ------------------------------------------------------------- naming ---
    def resolve(self, name: str) -> str | None:
        """The canonical id for a card-art filename, or None if it is not recognised.

        Punctuation is normalised away, so `ninomae_ina'nis` matches `ninomae_inanis`
        without needing an entry in the alias table -- and so will any future name whose
        only difference is an apostrophe or a plus sign.

        Beyond that, aliases and then an exact match. **Deliberately no fuzzy matching.**
        One art file was once named `usada_pekore`, a single letter from a real holomem,
        and anything willing to close a one-letter gap would as readily map a genuine
        holomem onto the wrong one. Unrecognised is a useful answer that gets a filename
        fixed in one message; wrong is a holomem the reader can never name.
        """
        key = _normalise_id(name)
        key = _normalise_id(self.aliases.get(key, key))
        return self._normalised.get(key)

    @property
    def _normalised(self) -> dict[str, str]:
        return {_normalise_id(c): c for c in self.characters}

    def coverage(self, art_names: list[str]) -> tuple[dict[str, str], list[str], list[str]]:
        """What art resolves to, what does not, and which holomem have none.

        Three lists because they need three different responses: resolved art is usable,
        unresolved art is a filename to fix, and a holomem with no art is simply one the
        reader will refuse to name until a capture provides it.
        """
        resolved: dict[str, str] = {}
        unresolved: list[str] = []
        for name in art_names:
            canonical = self.resolve(name)
            if canonical:
                resolved[name] = canonical
            else:
                unresolved.append(name)
        missing = sorted(self.characters - set(resolved.values()))
        return resolved, sorted(unresolved), missing


# ------------------------------------------------------------------- panel ----
def count_members(panel: np.ndarray) -> list[int]:
    """How many real portraits each row of the panel holds.

    No recognition involved: unused cells carry a flat grey placeholder with a triangle,
    and a placeholder is the only thing in the grid with almost no colour and almost no
    detail. Counting these is what validates a group label, so it must not depend on the
    thing it validates.
    """
    height, width = panel.shape[:2]
    if height < PANEL_ROWS or width < PANEL_COLUMNS:
        raise PanelError(f"panel crop is too small: {width}x{height}")

    counts = []
    for row in range(PANEL_ROWS):
        top, bottom = height * row // PANEL_ROWS, height * (row + 1) // PANEL_ROWS
        filled = 0
        for column in range(PANEL_COLUMNS):
            left = width * column // PANEL_COLUMNS
            right = width * (column + 1) // PANEL_COLUMNS
            if not _is_placeholder(panel[top:bottom, left:right]):
                filled += 1
        counts.append(filled)
    return counts


def _is_placeholder(cell: np.ndarray) -> bool:
    if cell.size == 0:
        return True
    pixels = cell.astype(np.float32)
    # Sample the middle, away from the grid lines between cells.
    h, w = pixels.shape[:2]
    inner = pixels[h // 6:h - h // 6, w // 6:w - w // 6]
    if inner.size == 0:
        inner = pixels

    high, low = inner.max(axis=2), inner.min(axis=2)
    saturation = float(np.mean(np.where(high > 0, (high - low) / np.maximum(high, 1.0), 0.0)))
    detail = float(inner.mean(axis=2).std())
    return saturation <= PLACEHOLDER_MAX_SATURATION and detail <= PLACEHOLDER_MAX_DETAIL


def panel_is_readable(counts: list[int], book: GroupBook) -> bool:
    """Whether these member counts could have come from a panel at all.

    Every hololive branch has 3, 4 or 5 members, so a row counted as anything else means
    the panel is not currently readable rather than that the game dealt an odd group. In
    practice that means a payout is being displayed: its panels sit over the grid, and a
    frame mid-payout counted [4, 3, 2, 2] where the clean frames either side of it both
    counted [4, 4, 4, 5].

    A free check, needing nothing the reader does not already measure -- and the reader has
    to know when it cannot see, because the alternative is advising from half a roster.
    """
    if len(counts) != PANEL_ROWS:
        return False
    sizes = {group.size for group in book.groups}
    return all(count in sizes for count in counts)


def roster_from_groups(
    book: GroupBook,
    group_ids: list[str],
    *,
    member_counts: list[int] | None = None,
    bonus: str | None = None,
) -> ObservedRoster:
    """The roster implied by four group labels, checked against the panel's own counts.

    `member_counts` comes from `count_members` and is the whole point: it is measured
    without recognising anything, so it is independent evidence about the labels. A
    mismatch means a label was misread, and the only safe response is to refuse.
    """
    if member_counts is not None and len(member_counts) != len(group_ids):
        raise PanelError(
            f"read {len(group_ids)} labels but {len(member_counts)} rows of members"
        )

    groups: list[ObservedGroup] = []
    for index, group_id in enumerate(group_ids):
        group = book.by_id(group_id)
        if member_counts is not None and member_counts[index] != group.size:
            raise PanelError(
                f"row {index + 1} reads as {group.label} with {group.size} members, "
                f"but the panel shows {member_counts[index]} -- the label is misread, "
                f"or {group.label}'s membership in hololive_groups.yaml is wrong"
            )
        groups.append(
            ObservedGroup(id=group.id, name=group.label, members=group.members)
        )

    return ObservedRoster(groups=tuple(groups), bonus_character=bonus)


def read_roster(
    panel: np.ndarray,
    labels: np.ndarray,
    *,
    book: GroupBook,
    reader: "LabelReader",
    bonus: str | None = None,
) -> ObservedRoster:
    """The whole roster, from the panel grid and the badges beside it.

    Three independent things have to agree before this returns, and it raises rather than
    degrades if any of them does not:

    1. the member counts are all plausible group sizes, which is what tells us the panel is
       not currently hidden under a payout display;
    2. every badge is read confidently -- a partial roster is not a roster, since a missing
       group is four or five characters the deck contains and the agent does not know about;
    3. each badge's known size matches the count in its own row.

    The third is the one that earns its place. The first two are the reader agreeing with
    itself; only the count is measured without recognising anything, so only the count can
    contradict a confident misread.
    """
    counts = count_members(panel)
    if not panel_is_readable(counts, book):
        raise PanelError(
            f"member counts {counts} are not four real group sizes -- the panel is "
            f"covered, most likely by a payout display"
        )

    read = reader.read(labels)
    refused = [(index, label) for index, label in enumerate(read) if not label.confident]
    if refused:
        detail = "; ".join(f"row {i + 1}: {label.reason}" for i, label in refused)
        raise PanelError(f"could not read every badge -- {detail}")

    return roster_from_groups(
        book,
        [book.by_badge(label.badge).id for label in read],
        member_counts=counts,
        bonus=bonus,
    )


def read_table_roster(
    frame: np.ndarray,
    area,
    *,
    book: GroupBook | None = None,
    reader: "LabelReader | None" = None,
    bonus: str | None = None,
) -> ObservedRoster:
    """`read_roster` against a whole captured frame, using the table layout.

    Only the table layout. The game also shows a "Groups coming up" screen before the deal
    that presents the same four rows much larger and in a different place; this will not
    read it, and must not be pointed at it. Its boxes land on felt there, so the badges
    refuse and nothing wrong is returned -- but the panel box happens to count [5, 5, 5, 5],
    which is four legal group sizes, so the count check alone would have waved it through.
    The badges refusing is the only thing standing between that screen and a fabricated
    roster, which is worth knowing before anyone relaxes a threshold.
    """
    from . import layout
    from .group_labels import LabelReader

    book = book or GroupBook.load()
    reader = reader if reader is not None else LabelReader.load()
    return read_roster(
        area.crop(frame, layout.GROUP_PANEL),
        area.crop(frame, layout.GROUP_LABELS),
        book=book, reader=reader, bonus=bonus,
    )
