"""Turning what the screen reader has accumulated into the state the agents already speak.

This is the seam the whole of M8 was built toward. `agents/advisor.py` says it plainly: "at M8
the screen reader becomes another source of `PublicState` feeding the same call, and nothing
here changes." Nothing here changes it.

**The roster is per round, so the card space is too.** `rules/pokajan_v1.yaml` does not describe
a catalogue of every holomem the game knows -- it describes *one* round, four groups and
seventeen characters, because a slot index only means anything relative to the roster in play.
So a round read off the screen needs its own `Rules` built from the four groups it actually
showed, and `round_rules` builds one by substituting the roster into the file's own blocks
rather than by editing anything on disk.

**Everything unread is short, never guessed.** `table` and `scored` come back as *lower bounds*
-- the cards actually seen -- and that is a deliberate choice about which way to be wrong.
`envs/belief.py` computes `hand + table + scored` as "cards accounted for" and treats the
remainder as unseen, so under-counting makes the belief think more cards are still out there.
That is the safe direction: it makes the agent less certain than it could be, where an
over-count would make it confidently wrong about cards it has already watched leave.

Two seats' discard staircases are unread and about four turns in five pass unobserved, so the
shortfall is real and it is stated: `PublicView.short_by` is how many cards conservation says
are on the table that no reader has named.

**What is still missing is missing, and named.** `phase`, `last_discard` and `turn_index` have
no reader, and `public_view` reports that rather than inventing them. `current_seat` does have
one now -- the four bars around the central oval -- but it is covered on every payout frame, so
what comes back is the last frame that could see it and `turn_read_at` says when that was.

None of the three gaps blocks the overlay's main question. "What should I throw, and what is
dangerous" is a property of the position, not of whose turn it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from ..core.rules import Rules
from ..server.protocol import PublicState, SeatView
from .accumulate import RoundBelief, card_key
from .reader import Card
from .roster_panel import GroupBook

RULES_FILE = Path(__file__).resolve().parents[2] / "rules" / "pokajan_v1.yaml"

# Which seat the reader is looking out of. The capture is of this machine's screen, so the
# hand at the bottom of the table is the player's own and every other seat is an opponent.
VIEWER_SEAT = "bottom"


def round_rules(roster: Iterable[str], *, book: GroupBook | None = None,
                base: Path = RULES_FILE) -> Rules:
    """`Rules` whose card space is exactly the four groups this round deals.

    Built by substituting the roster into the rules file's own blocks, so every number that is
    not the roster -- the payout table, the hand limit, the deck size, the coin floor -- comes
    from the one place gameplay numbers are allowed to live, and a corrected payout stays a
    one-line edit there rather than something to be mirrored here.
    """
    import yaml

    book = book or GroupBook.load()
    by_id = {group.id: group for group in book.groups}
    wanted = list(roster)
    missing = [gid for gid in wanted if gid not in by_id]
    if missing:
        raise KeyError(f"roster names groups the book does not have: {missing}")

    raw = yaml.safe_load(base.read_text(encoding="utf-8"))
    names = {entry["id"]: entry.get("name", entry["id"]) for entry in raw["characters"]}

    members: list[str] = []
    for gid in wanted:
        for holomem in by_id[gid].members:
            if holomem not in members:
                members.append(holomem)

    # Display names exist only for the seventeen holomem the rules file's own roster happens to
    # deal, so most rounds mix "Gawr Gura" with "natsuiro_matsuri" and the advice reads like a
    # bug. The id is the fallback rather than a title-cased guess on purpose: several of these
    # are stylised -- `azki` is "AzKi", `irys` is "IRyS", `ninomae_inanis` keeps an apostrophe --
    # and a confident wrong name is worse than an obviously mechanical one for a human trying to
    # find the card on screen. `groups.yaml` carries no names to take them from.
    raw["characters"] = [{"id": h, "name": names.get(h) or h.replace("_", " ")}
                         for h in members]
    raw["groups"] = [{"id": gid, "name": by_id[gid].label,
                      "members": list(by_id[gid].members)} for gid in wanted]
    # The file's own bonus is a fixture for the engine's tests; a live round reads its own.
    raw["bonus_character"] = None
    return Rules.from_dict(raw, path=f"{base} + roster {'/'.join(wanted)}")


@dataclass(frozen=True)
class PublicView:
    """A `PublicState` the agents can consume, plus an honest account of what is missing.

    `state` is None exactly when the round cannot be advised on at all. When it is present the
    numbers in it are real, but `short_by` and `unknown` say how much of the table went unseen
    and which fields had no reader -- so a caller can show advice and its caveats together
    rather than either hiding the gap or refusing over it.
    """

    state: PublicState | None
    reasons: tuple[str, ...] = ()
    short_by: int = 0
    unknown: tuple[str, ...] = ()
    # When `state.current_seat` was last actually seen, and None if no frame in the round ever
    # showed a lit bar -- in which case the field holds the viewer as a placeholder and means
    # nothing. A payout covers the indicators, so this is routinely a turn or two old.
    turn_read_at: float | None = None
    # How many of the player's own cards the reader never named. They are still *held* -- the
    # advice is sound, just built on a narrower hand -- but the count vector is short by this
    # much, so the recommendation is priced as if the hand were smaller than it is.
    hand_unread: int = 0


# Fields of `PublicState` that no reader produces yet, with what each would take. Listed once,
# here, so the overlay can say what it does not know instead of showing a confident default.
UNREAD_FIELDS = {
    "phase": "draw and discard are distinguishable only by whether the seat is holding eight "
             "cards, and only the viewer's own hand is countable",
    "last_discard": "the newest card in each field, which is the reader `data/pending/` was "
                    "collected for and which does not exist yet",
    "turn_index": "nothing counts turns independently; the deck counter cannot stand in "
                  "because a claim does not move the turn while a chained call draws several "
                  "times without one",
}


class RoundAdvisor:
    """Advice from a live round, with the per-round setup done once.

    Two things are cached because they are per *round* and not per frame. Building a `Rules` for
    a roster re-parses the rules file and costs 19 ms against the 9 ms the advice itself takes,
    so rebuilding it every frame would make the setup twice the work. And `Advisor` accumulates
    a belief across the states it is shown -- `agents/advisor.py` is explicit that an unclaimed
    discard is only visible as a difference between two consecutive views -- so a fresh one per
    frame would be blind to exactly the signal the belief is built on.

    Keyed on the roster, which is what changes when the round does.
    """

    def __init__(self, *, particles: int = 48, seed: int | None = None,
                 book: GroupBook | None = None) -> None:
        self.particles = particles
        self.seed = seed
        self.book = book
        self._rules: dict[tuple[str, ...], Rules] = {}
        self._advisors: dict[tuple[str, ...], object] = {}

    def rules_for(self, roster: tuple[str, ...]) -> Rules:
        if roster not in self._rules:
            self._rules[roster] = round_rules(roster, book=self.book)
        return self._rules[roster]

    def advise(self, belief: RoundBelief, *, game_id: str = "live"):
        """`(view, recommendation)`, with the recommendation None when there is nothing to say.

        The view comes back either way, because "no advice, and here is why" is the answer the
        overlay needs on the frames that cannot be read -- silence looks identical to a reader
        that has crashed.
        """
        from ..agents.advisor import Advisor
        from ..server.protocol import DecisionRequest

        view = public_view(belief, book=self.book, game_id=game_id)
        if view.state is None:
            return view, None

        roster = tuple(belief.roster or ())
        rules = self.rules_for(roster)
        if roster not in self._advisors:
            self._advisors[roster] = Advisor(rules, seed=self.seed, particles=self.particles)
        advisor = self._advisors[roster]

        mask = [count > 0 for count in view.state.hand] + [False] * 3
        if not any(mask):
            return view, None
        return view, advisor.recommend(DecisionRequest(
            game_id=game_id, seat=view.state.viewer, decision="DISCARD",
            legal_mask=mask, state=view.state))


def public_view(belief: RoundBelief, *, book: GroupBook | None = None,
                game_id: str = "live") -> PublicView:
    """What the agents can be told about this round, and what they cannot."""
    tracking = belief.tracking
    if not tracking.ready:
        return PublicView(None, reasons=tracking.reasons, unknown=tuple(UNREAD_FIELDS))

    rules = round_rules(belief.roster or (), book=book)
    space = rules.cards
    viewer = belief.seats.index(VIEWER_SEAT)

    def vector(tally: Mapping[str, int]) -> list[int]:
        """A `{"gawr_gura:blue": 2}` tally as a slot count vector."""
        counts = space.zeros()
        for key, count in tally.items():
            name, _, colour = key.partition(":")
            try:
                counts[space.slot(name, colour)] += count
            except KeyError:
                # A card this round cannot deal is a misread. Dropping it keeps the vector a
                # lower bound rather than turning it into a wrong one.
                continue
        return counts

    def hand_vector(cards: Iterable[Card]) -> list[int]:
        tally: dict[str, int] = {}
        for card in cards:
            key = card_key(card)
            if key is not None:
                tally[key] = tally.get(key, 0) + 1
        return vector(tally)

    table = vector(belief.discards.counts())
    scored = vector(belief.scored)
    span = belief.table_span
    short_by = max(0, (span[0] if span else 0) - sum(table))

    coins = belief.ledger.latest
    seats = [
        SeatView(
            seat=index,
            coins=int(coins.get(seat, 0)),
            hand_size=rules.play.hand_limit,
            discards=vector(belief.discards.seen.get(seat, {})),
            calls_made=sum(1 for payout in belief.payouts if payout.winner == seat),
            coins_won=sum(payout.amount for payout in belief.payouts if payout.winner == seat),
            coins_paid=sum(payout.paid.get(seat, 0) for payout in belief.payouts),
        )
        for index, seat in enumerate(belief.seats)
    ]

    bonus = belief.bonus_holomem
    state = PublicState(
        game_id=game_id,
        turn_index=0,
        viewer=viewer,
        hand=hand_vector(belief.hand),
        seats=seats,
        deck_remaining=belief.deck_remaining or 0,
        table=table,
        scored=scored,
        last_discard_slot=None,
        last_discard_seat=None,
        # The last frame that could see the bars, which is not necessarily this one -- a payout
        # covers them. `PublicView.turn_read_at` says how old the answer is, and it falls back
        # to the viewer only when no frame in the round has ever shown a lit bar.
        current_seat=(belief.seats.index(belief.current_seat)
                      if belief.current_seat in belief.seats else viewer),
        phase="discard",
        finished=any(value <= 0 for value in coins.values()),
        final_coins=[int(coins.get(seat, 0)) for seat in belief.seats],
        coins_minted=belief.ledger.minted,
        bonus_character=space.char_index(bonus) if bonus else None,
        recent_discards=[[] for _ in belief.seats],
    )
    return PublicView(state, short_by=short_by, unknown=tuple(UNREAD_FIELDS),
                      turn_read_at=belief.current_seat_at,
                      hand_unread=belief.hand_unread or 0)
