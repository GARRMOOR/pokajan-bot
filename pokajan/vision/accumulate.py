"""What has happened, accumulated from a stream of what was on screen.

`reader.FrameReading` answers "what does this frame show". This module answers "what has
happened so far", which no frame shows and which therefore has to be remembered. Keeping the
two apart is the point: a reader that also remembers cannot tell a fresh observation from a
recalled one, and the failure that produces is confident advice about a game that has moved on.

Three things get accumulated here, and each one is shaped by a measurement rather than by what
would be convenient.

**Coins, into scoring events.** The coin displays are the most reliable thing on the table and
they are the only channel through which a payout is observable at all once the meld has left.
The catch is that the four seats are not read on the same frame -- one refuses while its
neighbour reads -- so a payout arrives piecemeal and a ledger that settles on the first
non-zero difference splits one event into two and never recovers. Measured across the two
complete rounds in `data/games/`: settling eagerly produced **8 unexplained events** and left
both rounds' books unbalanced. Settling only when the difference forms a *legal payout* --
one winner, one payer or three, shares equal, mint explained by a payer at zero -- produced
**21 events, every amount a legal entry in the payout table, and both rounds closing at exactly
4000 + minted**. So `CoinLedger` holds a candidate open until the books balance.

**The payout amount, into the shape of the hand that scored.** `scored` cannot be read directly
except during the payout animation, but the amount constrains it hard, because the payout table
is sparse and the bonus is additive per scoring copy. A triple's scoring set is three copies of
one holomem, so it carries 0 or 3 bonus copies and never 1; a group carries 0 or 1. That is what
made 930 decode uniquely to a monochrome four-group holding the bonus. `shapes_paying` returns
every shape that could have paid an amount, which is often one and never zero for a real payout.

**Discards, into a floor rather than a history.** This is the part where the obvious design
loses to a measurement. The intended plan was to read the newest discard once per turn and
append it, which requires seeing every turn. Over the two complete rounds the deck counter fell
one at a time on **17 of 81** draws and **11 of 59** -- about a fifth. Four turns in five pass
without an individual observation, so an append-per-turn history would be missing most of the
cards it claimed to hold, and worse, would not know which.

The replacement does not need per-turn continuity and cannot be wrong. Every single observation
of a discard row is a true subset of what that seat has thrown, so the largest count of a card
ever seen in one observation is a **lower bound** on how many of it are gone. That is all the
belief actually requires: a card seen on the table is definitely not in a hand and definitely
not in the deck, and a bound that is short simply carries less information rather than false
information. When the row never saturates -- short rounds, few discards -- the bound is exact.

Nothing here is allowed to quietly become uncertain. `Tracking` names what is intact and what
is not, and it is the accumulator's answer to the only question an advisor may ask of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping

from ..core.rules import HandKind, Rules
from . import layout
from .reader import Card, FrameReading, Meld

# How many coin observations a candidate payout may stay unbalanced before the ledger gives up
# and declares that it has lost the thread.
#
# Measured rather than picked: across the two complete rounds, every one of the 21 events
# balanced within **two** observations of opening, and most within one. Six is three times the
# worst case, which is enough slack for a slow payout animation and still short enough that a
# genuinely missed event is noticed within a turn rather than at the end of the round.
#
# What this bounds is the damage from *two* payouts landing without the state between them ever
# being seen. Their combined difference is usually not a legal single payout, so the ledger sits
# open and this is what turns that into an announcement instead of a permanent stall.
LEDGER_PATIENCE = 6

# How much agreement it takes to name the bonus holomem from reads that individually refused.
# See `RoundBelief.bonus_by_vote`. The live case that needed it ran 68 votes against a
# best-of-the-rest of 3 -- 23x -- so 10 frames and 3x is far inside the observed signal while
# still being well outside anything scattered noise produces.
BONUS_VOTE_MIN = 10
BONUS_VOTE_RATIO = 3

# How many of the player's own cards may still be unread before advice is refused.
#
# Requiring the *whole* hand on a single frame is the same fragility the bonus card had, and it
# is worse here because there are seven chances to fail rather than one. Measured across every
# logged round: a slot reads on 12% to 38% of frames in a bad round, so seven coinciding is
# vanishingly unlikely -- one 34-frame round read every slot at some point and never all seven
# at once, and produced no advice at all.
#
# Merging across frames fixes most of it (`_merge_hand`), and this absorbs the rest. It is a
# judgement rather than a measurement: a card that has not been read is definitely still *held*,
# so advice built on the rest is sound but narrower -- the risk is recommending a discard when an
# unread card was the better throw, or breaking a triple the reader cannot see. Two is where that
# stops being a caveat and starts being a guess. `PublicView.hand_unread` carries the number
# through so the panel can say so rather than implying a hand it has not seen.
HAND_UNREAD_MAX = 2


# ------------------------------------------------------------- what scored ----
@dataclass(frozen=True)
class Shape:
    """A hand shape that could have produced a given payout."""

    kind: HandKind
    group_size: int | None
    monochrome: bool
    bonus_copies: int

    @property
    def cards(self) -> int:
        """How many cards this shape removed from the table."""
        return 3 if self.kind is HandKind.TRIPLE else int(self.group_size or 0)

    def __str__(self) -> str:
        what = "triple" if self.kind is HandKind.TRIPLE else f"{self.group_size}-group"
        parts = [("mono " if self.monochrome else "") + what]
        if self.bonus_copies:
            parts.append(f"+{self.bonus_copies} bonus")
        return " ".join(parts)


def shapes_paying(amount: int, rules: Rules, *,
                  group_sizes: Iterable[int] | None = None) -> tuple[Shape, ...]:
    """Every hand shape that pays exactly `amount`, from the loaded payout table.

    `group_sizes` narrows the search to the groups actually on the table this round, which is
    worth passing whenever the roster has been read: it is what separates a five-group from a
    monochrome three-group when both would pay 480.

    The bonus counts are not a free range. The bonus is additive per copy *in the scoring set*,
    and a triple's scoring set is three copies of one holomem -- so it holds either all three or
    none, never one or two. A group holds one of each member, so it holds the bonus holomem once
    or not at all. Those two constraints are what make most amounts decode uniquely.
    """
    sizes = sorted(set(group_sizes)) if group_sizes is not None else sorted(rules._table_group)
    found: list[Shape] = []
    for monochrome in (False, True):
        for copies in (0, 3):
            if rules.payout(HandKind.TRIPLE, monochrome=monochrome,
                            bonus_copies=copies) == amount:
                found.append(Shape(HandKind.TRIPLE, None, monochrome, copies))
        for size in sizes:
            if size not in rules._table_group:
                continue
            for copies in (0, 1):
                if rules.payout(HandKind.GROUP, group_size=size, monochrome=monochrome,
                                bonus_copies=copies) == amount:
                    found.append(Shape(HandKind.GROUP, size, monochrome, copies))
    return tuple(found)


def meld_shape(cards: Iterable[Card], *, bonus: str | None = None,
               group_of: Mapping[str, str] | None = None,
               group_size: Mapping[str, int] | None = None) -> Shape | None:
    """The shape a face-up meld is, or None when those cards do not form a scoring hand.

    Read off the cards rather than inferred from the amount, so it is the independent half of
    the cross-check in `RoundBelief`: the coin displays say what was paid, this says what was
    held, and a meld only counts as confirmed when the two agree on a shape.

    None is returned freely -- for a half-read meld, for cards that are neither one holomem
    three times nor one of each member of a group, and for a group whose members do not all
    belong to it. A meld that cannot be described is not evidence of anything.
    """
    cards = tuple(cards)
    names = [card.character for card in cards]
    colours = {card.colour for card in cards}
    if not cards or any(name is None for name in names) or None in colours:
        return None
    monochrome = len(colours) == 1

    if len(set(names)) == 1:
        # A triple's scoring set is three copies of one holomem, so it is all bonus or none.
        return Shape(HandKind.TRIPLE, None, monochrome, 3 if names[0] == bonus else 0)

    if len(set(names)) != len(names) or group_of is None or group_size is None:
        return None
    groups = {group_of.get(name) for name in names}
    if len(groups) != 1:
        return None
    group = groups.pop()
    if group is None or group_size.get(group) != len(cards):
        return None                      # some member is missing, so the group is not complete
    return Shape(HandKind.GROUP, len(cards), monochrome, 1 if bonus in names else 0)


def _coin_granularity(rules: Rules) -> int:
    """The largest step every coin total is a multiple of, from the payout table itself."""
    from math import gcd

    amounts = set()
    for kind, size in ((HandKind.TRIPLE, None), (HandKind.GROUP, 3),
                       (HandKind.GROUP, 4), (HandKind.GROUP, 5)):
        for monochrome in (False, True):
            for copies in (0, 1, 3):
                try:
                    amounts.add(rules.payout(kind, group_size=size, monochrome=monochrome,
                                             bonus_copies=copies))
                except Exception:
                    continue
    amounts.discard(0)
    shares = {amount // 3 for amount in amounts if amount % 3 == 0}
    step = 0
    for value in list(amounts) + list(shares) + [rules.play.initial_coins]:
        step = gcd(step, int(value))
    return step or 1


@dataclass(frozen=True)
class Payout:
    """One scoring event, as reconstructed from the coin displays.

    `shapes` is what the amount permits, not what was seen. It is empty when the amount is not
    in the payout table at all, which is a loud signal: either the table is wrong or the ledger
    has merged two events, and both are worth stopping for.

    `meld` is the cards themselves, present only when a face-up meld was read on a nearby frame
    *and* its shape is one the amount permits. When it is present `shapes` has been narrowed to
    that one shape, so `cards_scored` becomes exact and everything downstream of it tightens.
    """

    at: float
    winner: str
    amount: int
    paid: Mapping[str, int]
    minted: int
    claimed: bool
    shapes: tuple[Shape, ...] = ()
    meld: tuple[Card, ...] = ()

    @property
    def cards_scored(self) -> int | None:
        """How many cards left the table, or None when the candidates disagree."""
        counts = {shape.cards for shape in self.shapes}
        return counts.pop() if len(counts) == 1 else None

    @property
    def payer(self) -> str | None:
        """The seat that paid alone, for a claimed hand. None for a split."""
        return next(iter(self.paid), None) if self.claimed else None

    def __str__(self) -> str:
        how = f"claimed off {self.payer}" if self.claimed else "self-drawn, split three ways"
        shapes = " or ".join(str(s) for s in self.shapes) or "no shape pays this"
        mint = f", {self.minted} minted" if self.minted else ""
        seen = f" [{' '.join(str(c) for c in self.meld)}]" if self.meld else ""
        return f"{self.winner} +{self.amount} ({how}{mint}) -- {shapes}{seen}"


# ------------------------------------------------------------- the ledger ----
class LedgerError(RuntimeError):
    """The coin displays stopped adding up."""


class CoinLedger:
    """Coins per seat, and the scoring events that moved them.

    Settles only on a **balanced** difference. Everything else is held open, because the seats
    are not read on the same frame and half a payout looks exactly like a small one.

    What counts as balanced is the payout rules, not arithmetic: exactly one seat gains; either
    one seat pays the whole amount or three pay an equal third; a payer may fall short only by
    stopping at zero, and any shortfall is minted. That is four separate rules checking each
    other on every event, which is why a misread coin display fails to settle rather than
    quietly becoming a wrong balance.
    """

    def __init__(self, rules: Rules, *, seats: Iterable[str] = layout.SEAT_ORDER,
                 patience: int = LEDGER_PATIENCE,
                 group_sizes: Iterable[int] | None = None,
                 coins: Mapping[str, int] | None = None,
                 minted: int = 0) -> None:
        self.rules = rules
        self.seats = tuple(seats)
        self.patience = patience
        # No coin total can be anything but a multiple of this, and it is a theorem rather than
        # a heuristic: every payout in the table divides by three, every three-way share is a
        # multiple of ten, the stack starts at a multiple of ten, and the bankruptcy floor stops
        # a payer at zero. Derived from the rules rather than written as 10, because the payout
        # table is the one thing here still expected to be corrected.
        #
        # It is worth having because a digit misread is otherwise indistinguishable from a real
        # coin change. Measured across every logged round: **4 readings of 8391** break it --
        # values of 1, 13, 4 and 1, all on covered or animating displays -- and one of them, a
        # `bottom` seat reading 1 as the game-over banner came up, put a whole 138-frame round
        # out of balance by exactly one coin.
        self.granularity = _coin_granularity(rules)
        # How many readings were thrown out per seat, so a box that has quietly started
        # misreading shows up as a number rather than as a round that will not settle.
        self.rejected: dict[str, int] = {}
        self.group_sizes = tuple(group_sizes) if group_sizes is not None else None
        # `coins` starts a ledger somewhere other than the deal, which is only ever correct
        # when the state is known from outside -- a test, or a resumed round whose earlier
        # events were accounted for elsewhere. It is not a way to recover from having lost
        # track: adopting a state the ledger cannot explain is precisely what the balance gate
        # exists to refuse, and `minted` has to come with it or the books will not close.
        self.coins = dict(coins) if coins else {seat: rules.play.initial_coins
                                                for seat in self.seats}
        self.latest = dict(self.coins)
        self.minted = minted
        self.payouts: list[Payout] = []
        self.lost: str | None = None
        self._open: int = 0

    # ------------------------------------------------------------------------
    @property
    def pending(self) -> bool:
        """Whether a difference is currently unexplained."""
        return self.latest != self.coins

    @property
    def total(self) -> int:
        return sum(self.latest.values())

    @property
    def balances(self) -> bool:
        """The accounting identity the whole engine is tested against."""
        return self.total == len(self.seats) * self.rules.play.initial_coins + self.minted

    def observe(self, coins: Mapping[str, int], *, at: float = 0.0) -> Payout | None:
        """Fold one frame's coin readings in. Returns an event if this frame completed one.

        Seats absent from `coins` keep their last known value, which is correct rather than
        merely convenient: coins only move on a payout, so a seat that could not be read has
        not changed unless the payout being assembled changed it -- and that case is exactly
        what holding the candidate open resolves.
        """
        if self.lost:
            return None
        for seat, value in coins.items():
            if seat in self.latest and value % self.granularity == 0:
                self.latest[seat] = value
            elif seat in self.latest:
                self.rejected[seat] = self.rejected.get(seat, 0) + 1

        if not self.pending:
            self._open = 0
            return None

        event = self._balance(at=at)
        if event is None:
            self._open += 1
            if self._open > self.patience:
                self.lost = (
                    f"coins moved from {self.coins} to {self.latest} and no legal payout "
                    f"explains it after {self._open} readings -- most likely two payouts "
                    f"landed without the state between them being seen"
                )
            return None

        self.payouts.append(event)
        self.minted += event.minted
        self.coins = dict(self.latest)
        self._open = 0
        return event

    # ------------------------------------------------------------------------
    def _balance(self, *, at: float) -> Payout | None:
        """The difference as a legal payout, or None while it is not one yet."""
        delta = {seat: self.latest[seat] - self.coins[seat] for seat in self.seats}
        winners = [seat for seat, d in delta.items() if d > 0]
        payers = [seat for seat, d in delta.items() if d < 0]
        # One winner, and either the discarder alone or the other three. Two winners is not a
        # bigger payout, it is two payouts seen as one.
        if len(winners) != 1 or len(payers) not in (1, len(self.seats) - 1):
            return None

        winner = winners[0]
        amount = delta[winner]
        claimed = len(payers) == 1
        share, remainder = divmod(amount, len(payers))
        if remainder:
            return None                      # a split that does not divide is not this game's

        paid: dict[str, int] = {}
        for seat in payers:
            owed = -delta[seat]
            if owed == share:
                paid[seat] = owed
                continue
            # Short only by stopping at the floor, and only if the display agrees it is there.
            if owed < share and self.latest[seat] == self.rules.end.coin_floor:
                paid[seat] = owed
                continue
            return None

        minted = amount - sum(paid.values())
        if minted and not self.rules.caller_gains_full_amount_on_payer_bankruptcy:
            return None
        return Payout(
            at=at, winner=winner, amount=amount, paid=paid, minted=minted, claimed=claimed,
            shapes=shapes_paying(amount, self.rules, group_sizes=self.group_sizes),
        )


# ------------------------------------------------------------ the discards ----
def card_key(card: Card) -> str | None:
    """`character:colour`, or None when either half was refused.

    A half-read card is dropped rather than recorded as a partial. A count vector has no way to
    express "some pink card", and the alternative -- guessing the missing half -- puts a card
    the belief will treat as certain into a slot nothing ever saw.
    """
    return f"{card.character}:{card.colour}" if card.known else None


@dataclass
class DiscardFloor:
    """A lower bound on what each seat has thrown, merged from overlapping views.

    Never an exact history, and the docstring at the top of this module has the measurement
    that decided it: four turns in five go unobserved, and the near end of a discard pile
    buries its oldest cards, so an appended history would be short by an unknown amount in an
    unknown place. This is short by an unknown amount in a *known direction*, which the belief
    can use safely.

    The merge is `max` per card per seat over every observation, which needs no alignment
    between views and so cannot be defeated by a seat throwing the same card twice. Sequence
    alignment was the alternative and it is ambiguous exactly there -- a row reading
    `watame, watame, gura` overlaps a later `watame, gura, polka` in two different ways.
    """

    seats: tuple[str, ...] = layout.SEAT_ORDER
    seen: dict[str, dict[str, int]] = field(default_factory=dict)
    unread: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.seen = {seat: {} for seat in self.seats}
        # The two seats either side lay their discards out as sheared diagonal staircases, so
        # `find_row` does not describe them and `TableReader.read_discards` refuses them. They
        # are named here from the start so the shortfall is a stated hole rather than a silence.
        self.unread = {seat for seat in self.seats if seat not in layout.DISCARD_ASPECT}

    def observe(self, seat: str, cards: Iterable[Card]) -> None:
        """Fold in one view of one seat's discard field."""
        counts: dict[str, int] = {}
        for card in cards:
            key = card_key(card)
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
        known = self.seen.setdefault(seat, {})
        for key, count in counts.items():
            known[key] = max(known.get(key, 0), count)
        self.unread.discard(seat)

    def counts(self) -> dict[str, int]:
        """Every seat's floor summed: the table's lower bound as a count vector would hold it."""
        total: dict[str, int] = {}
        for seat in self.seen:
            for key, count in self.seen[seat].items():
                total[key] = total.get(key, 0) + count
        return total

    @property
    def cards(self) -> int:
        return sum(self.counts().values())


# ---------------------------------------------------------------- tracking ----
@dataclass(frozen=True)
class Tracking:
    """Whether what has been accumulated can be advised on, and what is missing if not.

    `ready` is about the things this accumulator claims to know exactly -- the roster, the
    bonus, the coins and the deck. It is deliberately **not** about `table`, which is a
    declared lower bound and would otherwise hold `ready` false for the whole round and teach
    a reader to ignore it. `table_complete` carries that separately, and it is false today for
    a reason that is named rather than implied.
    """

    ready: bool
    reasons: tuple[str, ...]
    table_complete: bool
    table_reasons: tuple[str, ...]

    def __str__(self) -> str:
        if self.ready:
            return "tracking: everything readable is read"
        count = f"{len(self.reasons)} reason" + ("s" if len(self.reasons) > 1 else "")
        return f"LOST TRACK -- no advice, {count}"


class RoundBelief:
    """Everything one round has shown, accumulated frame by frame.

    Fed `FrameReading`s in the order they were read, from any source: live capture or a replayed
    journal. Both produce the same object, which is what makes a logged round a test case for
    the advisor rather than only a record of one.

    Facts that are settled once and then defended: the roster and the bonus holomem do not
    change within a round, so a frame disagreeing with an established one is a misread and is
    refused rather than believed. Facts that move -- coins, deck, hand -- take the newest
    confident reading.
    """

    def __init__(self, rules: Rules, *, seats: Iterable[str] = layout.SEAT_ORDER) -> None:
        self.rules = rules
        self.seats = tuple(seats)
        self.roster: tuple[str, ...] | None = None
        self.bonus: str | None = None
        self.deck_remaining: int | None = None
        self.hand: tuple[Card, ...] = ()
        self.ledger = CoinLedger(rules, seats=self.seats)
        self.discards = DiscardFloor(seats=self.seats)
        self.frames = 0
        self.conflicts: list[str] = []
        self.scored: dict[str, int] = {}
        self.bonus_votes: dict[str, int] = {}
        # Outright reads of the bonus card, tallied rather than latched -- see `_settle_bonus`.
        self.bonus_reads: dict[str, int] = {}
        # Whose turn it was on the last frame that could tell. Kept rather than required,
        # because the bars are covered on every payout frame -- which is most of what the
        # watcher deliberately samples hardest -- and a stale answer one turn old is far more
        # use to an overlay than no answer at all. `current_seat_at` is how stale.
        self.current_seat: str | None = None
        self.current_seat_at: float | None = None
        # The deck reading the hand candidate belongs to. A draw changes the hand, so a moved
        # counter is the one signal that the candidate is describing a position that is gone.
        self._hand_deck: int | None = None
        self._book = None
        # The last frame's meld candidates, held because the two channels are not in step: the
        # meld is drawn while the payout animates and the coins finish moving a frame or two
        # later. Holding them is not the same as trusting them -- a candidate is only ever
        # attached to an event it was found in the winning seat's position for, whose amount
        # also permits its shape.
        self._melds: tuple[Meld, ...] = ()

    # ------------------------------------------------------------------------
    def observe(self, reading: FrameReading) -> Payout | None:
        """Fold one frame in. Returns a scoring event if this frame completed one."""
        self.frames += 1

        if reading.roster is not None:
            if self.roster is None:
                self.roster = reading.roster
                self.ledger.group_sizes = self._group_sizes()
            elif reading.roster != self.roster:
                self.conflicts.append(
                    f"roster read as {reading.roster} having settled on {self.roster}"
                )
        if reading.bonus_best is not None:
            self.bonus_votes[reading.bonus_best] = self.bonus_votes.get(
                reading.bonus_best, 0) + 1
        if reading.bonus is not None:
            self.bonus_reads[reading.bonus] = self.bonus_reads.get(reading.bonus, 0) + 1
        self.bonus = self._settle_bonus()

        if reading.current_seat is not None:
            self.current_seat = reading.current_seat
            self.current_seat_at = reading.at
        if reading.deck_remaining is not None:
            self.deck_remaining = reading.deck_remaining
        self._merge_hand(reading)
        for seat, cards in reading.discards.items():
            self.discards.observe(seat, cards)
        if reading.melds:
            self._melds = reading.melds

        payout = self.ledger.observe(reading.coins, at=reading.at)
        return self._confirm(payout) if payout is not None else None

    # ------------------------------------------------------------------------
    def _confirm(self, payout: Payout) -> Payout:
        """Attach a face-up meld to a settled payout, if one of the candidates fits it.

        Two channels that share no machinery: the coin displays say who was paid and how much,
        the cards say what was held. A candidate has to survive both -- it must have been found
        in the *winning seat's* position, and its shape must be one the amount permits.

        That second test does real work rather than rubber-stamping the first. A meld's box is
        right-anchored, so a shorter box also fits inside a longer meld: a four-card group call
        produces a three-card candidate as well, made of its rightmost three cards. Three of a
        four-member group is not a complete group, `meld_shape` returns None for it, and the
        subset is thrown out -- which is what happened the first time a group was called.

        Two survivors attach nothing. Nor does disagreement overrule the amount: the ledger has
        four rules cross-checking every event and the meld has a box measured from a handful of
        frames, so the newer claim yields.
        """
        candidates, self._melds = self._melds, ()
        fits = []
        for candidate in candidates:
            if candidate.seat != payout.winner:
                continue
            shape = meld_shape(candidate.cards, bonus=self.bonus_holomem, group_of=self._group_of(),
                               group_size=self._group_size())
            if shape is not None and shape in payout.shapes:
                fits.append((shape, candidate.cards))
        if len(fits) != 1:
            return payout
        shape, meld = fits[0]
        confirmed = replace(payout, shapes=(shape,), meld=meld)
        for card in confirmed.meld:
            key = card_key(card)
            if key is not None:
                self.scored[key] = self.scored.get(key, 0) + 1
        self.ledger.payouts[-1] = confirmed
        return confirmed

    # ------------------------------------------------------------------------
    @property
    def payouts(self) -> list[Payout]:
        return self.ledger.payouts

    def _merge_hand(self, reading: FrameReading) -> None:
        """Fold this frame's hand into the one being assembled, or start again.

        The hand cannot be counted on to arrive complete on any one frame -- seven cards, each
        refusing independently -- so it is built up across frames instead. What makes that safe
        is that a merge is only allowed when **every position both frames read agrees**. A hand
        that has changed shows up as a disagreement, and disagreement throws the candidate away
        rather than blending two hands into one that never existed.

        That check is doing real work: measured across consecutive frames with the deck counter
        unmoved, 3.85% of shared positions disagree, because the hand re-sorts when a card is
        inserted and every card after it shifts along. Merging positionally without the check
        would fabricate a card in one merge in twenty-six.

        A deck counter that has moved means a draw, which means the hand has changed, so the
        candidate is dropped without even looking.
        """
        hand = reading.hand
        if not hand:
            return
        if (reading.deck_remaining is not None and self._hand_deck is not None
                and reading.deck_remaining != self._hand_deck):
            self.hand = ()
        if reading.deck_remaining is not None:
            self._hand_deck = reading.deck_remaining

        if len(self.hand) != len(hand):
            self.hand = hand
            return
        if any(old.known and new.known and str(old) != str(new)
               for old, new in zip(self.hand, hand)):
            self.hand = hand
            return
        self.hand = tuple(old if old.known else new for old, new in zip(self.hand, hand))

    @property
    def hand_unread(self) -> int | None:
        """How many of the player's own cards are still unnamed, or None if none were seen."""
        if not self.hand:
            return None
        return sum(1 for card in self.hand if not card.known)

    def _settle_bonus(self) -> str | None:
        """The bonus holomem from every outright read of it, not from whichever came first.

        Latching on the first read and calling every later disagreement a conflict is the
        obvious spelling and it throws away rounds. One real round read the card outright
        exactly **twice** across 141 frames -- once `himemori_luna`, once `tokoyami_towa` --
        and the single later misread contradicted the single earlier good one, so the round
        went unadvisable. The evidence was never close: 98 frames ranked `himemori_luna` top
        against 9 for `tokoyami_towa`.

        The card does not change during a round, so two disagreeing reads mean one of them is
        wrong, and the tally of best-guesses across the whole round is what knows which. A
        vote that lands on a holomem *nobody ever read outright* does not settle it, though --
        that is two channels disagreeing rather than one of them being noisy, and it stays
        unresolved for `_confirm` to report.
        """
        if not self.bonus_reads:
            return None
        if len(self.bonus_reads) == 1:
            return next(iter(self.bonus_reads))
        agreed = self.bonus_by_vote
        return agreed if agreed in self.bonus_reads else None

    @property
    def bonus_holomem(self) -> str | None:
        """The bonus, however it was established: read outright, or agreed across the round."""
        return self.bonus if self.bonus is not None else self.bonus_by_vote

    @property
    def bonus_by_vote(self) -> str | None:
        """The bonus holomem when no frame could name it but the whole round agrees.

        **Not a lowered threshold -- a different and larger body of evidence.** The bonus card
        is the same card on every frame of a round, so dozens of reads of it are dozens of
        independent samples of one question, and `templates.Match.best` records the top-ranked
        holomem even when the score fell short. A live round supplied the case: `amelia_watson`
        at 0.42 against a 0.45 floor on **68 of 86 frames**, with the other 18 scattered one
        apiece across holomem that were not even in the roster. Pale blonde art on a pale ground
        gives cross-correlation little to grip -- the same reason `hakui_koyori` tops out at
        0.54 -- so a correct read can sit under the floor for the whole round, and without this
        the round is unadvisable from the first frame to the last.

        Deliberately strict, because being wrong here mis-prices every hand in the round. The
        leader must clear `BONUS_VOTE_MIN` frames, must beat the runner-up by
        `BONUS_VOTE_RATIO`, and must be a holomem this round can actually deal. Noise does not
        do that: it scatters, which is precisely what the other 18 frames did.
        """
        # A pure reading of the tally, with no reference to `self.bonus` -- `_settle_bonus`
        # calls this precisely *when* an outright read exists, to decide which of two
        # disagreeing ones to believe. Short-circuiting on `self.bonus` here made this return
        # None exactly when it was needed, and worse, unset a bonus that was already settled.
        if not self.bonus_votes:
            return None
        ranked = sorted(self.bonus_votes.items(), key=lambda kv: -kv[1])
        (leader, votes), runner_up = ranked[0], (ranked[1][1] if len(ranked) > 1 else 0)
        members = set(self._group_of())
        if votes < BONUS_VOTE_MIN or votes < BONUS_VOTE_RATIO * max(runner_up, 1):
            return None
        return leader if not members or leader in members else None

    @property
    def scored_cards(self) -> int | None:
        """How many cards have left the table in scored hands, or None if any is ambiguous.

        Counted from the payout amounts rather than from anything seen, because the meld is on
        screen only during the payout animation and leaves the table for good afterwards.
        """
        totals = [payout.cards_scored for payout in self.payouts]
        return None if any(count is None for count in totals) else sum(totals)  # type: ignore[arg-type]

    @property
    def scored_span(self) -> tuple[int, int]:
        """The narrowest range `scored` can be in, given every payout's candidate shapes.

        Worth having even when `scored_cards` is exact, and the only useful answer when it is
        not: a monochrome triple and a monochrome four-group both pay 840, so an unread meld
        leaves the count one card wide. The span is what the belief can still subtract from the
        deck with certainty at its low end.
        """
        low = high = 0
        for payout in self.payouts:
            sizes = [shape.cards for shape in payout.shapes]
            low += min(sizes) if sizes else 0
            high += max(sizes) if sizes else 0
        return low, high

    @property
    def table_span(self) -> tuple[int, int] | None:
        """How many cards are on the table, from card conservation rather than from reading it.

        This is the strongest thing the accumulator knows about `table`, and it arrives without
        a single discard being read. Every card is in exactly one place, so

            table = deck_size - deck_remaining - cards_in_hands - scored

        and three of those four are known: the deck counter is read directly, the payout ledger
        bounds `scored` through the shapes that pay each amount, and hands are at the limit
        except for the one seat sitting between its draw and its discard -- which is worth
        exactly one card of width, not an unknown.

        It bounds the *count*, never the identities, so it does not replace reading the discard
        fields. It does something the fields cannot: it stays exact through the seats that are
        unreadable and through every turn that passes unobserved, and it is what tells the
        discard floor when the floor has gone wrong. A floor holding more cards than the table
        can contain is a misread that would otherwise be invisible.

        None until the deck counter has been read, and None again whenever the ledger's books
        do not balance. That second refusal is the one that matters: a missed payout makes
        `scored` too small, which makes this too *large* by the same amount, and a table
        believed roomier than it is would let the cross-check below wave through the misread it
        exists to catch. The 6-frame log in `data/games/` is exactly that case -- it joins a
        round already in progress, never sees the payouts that built the coin totals it can
        read, and would otherwise claim 51 cards on a table holding nothing like that many.
        """
        if self.deck_remaining is None or not self.ledger.balances or self.ledger.lost:
            return None
        seats = len(self.seats)
        limit = self.rules.play.hand_limit
        # Hands are at the limit *except* for two seats, and both exceptions are worth exactly
        # as much width as they can be. One seat may be holding its drawn card before it
        # discards, which is the `draws_per_turn` above the limit.
        #
        # And the seat that called last may not have refilled yet, which is the same width
        # below. Scoring cards leave the hand first and the deck replaces them after, so a
        # caller sits at `limit - k` in between -- and when the call is the one that bankrupts
        # somebody, the round ends there and the refill never happens at all. Assuming full
        # hands made `table` too small by up to five cards, which is not a harmless error: the
        # cross-check below then reads a correct discard floor as a misread. Two real rounds
        # were lost that way, both of them ending on a four-card call that bankrupted a seat.
        #
        # Only the most recent call can be outstanding. Chaining refills to the limit before
        # another call is legal, so there is never a second unrefilled one behind it.
        outstanding = max((max(shape.cards for shape in payout.shapes)
                           for payout in self.payouts[-1:] if payout.shapes), default=0)
        in_hands_low = seats * limit - outstanding
        in_hands_high = seats * limit + self.rules.play.draws_per_turn
        scored_low, scored_high = self.scored_span
        rest = self.rules.deck_size - self.deck_remaining
        return (max(0, rest - in_hands_high - scored_high), max(0, rest - in_hands_low - scored_low))

    @property
    def table_floor(self) -> int:
        """The fewest cards the table can hold, from the discard fields.

        Not simply `discards.cards`. **A claimed call takes its card off a discard pile**, and
        `DiscardFloor` merges by `max` and never removes anything -- deliberately, since it has
        no way to tell a card that was claimed from one buried by a later discard. So a claimed
        card goes on being counted as sitting on the table while `scored` counts it too, and the
        same card is in two places at once.

        Measured across the logged rounds: **42 cards in 15 rounds** appear in both the floor
        and a meld that scored. Each claimed payout removed exactly one card from a pile, and
        which one does not matter here because this bounds the count and not the identities.
        """
        claimed = sum(1 for payout in self.payouts if payout.claimed)
        return max(0, self.discards.cards - claimed)

    def _groups(self):
        """The four groups in play, as catalogue entries. Empty until the roster is read."""
        if self.roster is None:
            return ()
        if self._book is None:
            from .roster_panel import GroupBook

            try:
                self._book = GroupBook.load()
            except Exception:                # a missing catalogue must not stop the ledger
                self._book = False
        if not self._book:
            return ()
        by_id = {group.id: group for group in self._book.groups}
        found = [by_id[gid] for gid in self.roster if gid in by_id]
        return tuple(found) if len(found) == len(self.roster) else ()

    def _group_sizes(self) -> tuple[int, ...] | None:
        """Sizes of the four groups in play, for narrowing a payout to one shape."""
        groups = self._groups()
        return tuple(len(g.members) for g in groups) if groups else None

    def _group_of(self) -> dict[str, str]:
        """Which of the four groups each holomem on the table belongs to."""
        return {member: group.id for group in self._groups() for member in group.members}

    def _group_size(self) -> dict[str, int]:
        return {group.id: len(group.members) for group in self._groups()}

    @property
    def tracking(self) -> Tracking:
        reasons: list[str] = []
        if self.roster is None:
            reasons.append("the roster has not been read, so no card can be named")
        if self.bonus is None and self.bonus_by_vote is None:
            reasons.append("the bonus holomem has not been read, so no hand can be priced")
        if self.deck_remaining is None:
            reasons.append("the deck counter has not been read")
        unread = self.hand_unread
        if unread is None:
            reasons.append("no frame has shown the hand at all")
        elif unread > HAND_UNREAD_MAX:
            reasons.append(
                f"{unread} of {len(self.hand)} cards in hand are still unread, more than the "
                f"{HAND_UNREAD_MAX} a recommendation can absorb"
            )
        if self.ledger.lost:
            reasons.append(self.ledger.lost)
        elif self.ledger.pending:
            reasons.append(
                f"a coin change is unexplained: {self.ledger.coins} -> {self.ledger.latest}"
            )
        if not self.ledger.balances:
            reasons.append(
                f"coins total {self.ledger.total} against "
                f"{len(self.seats) * self.rules.play.initial_coins + self.ledger.minted} "
                f"expected from {self.ledger.minted} minted"
            )
        for payout in self.payouts:
            if not payout.shapes:
                reasons.append(f"no hand shape pays {payout.amount}, which {payout.winner} won")
        if len(self.bonus_reads) > 1 and self.bonus is None:
            # Recomputed rather than recorded when it happened, because unlike a roster
            # disagreement this one can *resolve*: another dozen frames ranking one of them top
            # settles it, and a round should not stay unadvisable for a misread it has since
            # outvoted. Only a genuine stalemate reaches here.
            tally = ", ".join(f"{name} x{count}" for name, count in
                              sorted(self.bonus_reads.items(), key=lambda kv: -kv[1]))
            reasons.append(
                f"the bonus card read as {tally} on different frames and the round's "
                f"{sum(self.bonus_votes.values())} rankings do not settle which"
            )
        span = self.table_span
        if span is not None and self.table_floor > span[1]:
            # Two independent channels disagreeing about the same quantity, and neither is
            # automatically the liar. Conservation rests on the deck counter and the payout
            # amounts; the floor rests on reading the discard rows. What this used to say --
            # "the discard fields were misread, most likely a row segmented into too many
            # cards" -- was wrong twice over, and both errors were on the conservation side:
            # claimed cards were still being counted as sitting on the table, and hands were
            # assumed full when the last caller had not refilled. Both are fixed above, so a
            # complaint here now means something genuinely unaccounted for.
            reasons.append(
                f"the discard fields show {self.table_floor} cards on the table but "
                f"conservation allows at most {span[1]}"
            )
        reasons.extend(self.conflicts)

        table_reasons: list[str] = []
        if self.discards.unread:
            table_reasons.append(
                f"{', '.join(sorted(self.discards.unread))} lay their discards out as diagonal "
                f"staircases and are not read at all"
            )
        table_reasons.append(
            "the rest is a lower bound, not a history -- about four turns in five pass without "
            "being seen and a pile buries its oldest cards"
        )
        return Tracking(
            ready=not reasons,
            reasons=tuple(reasons),
            table_complete=False,
            table_reasons=tuple(table_reasons),
        )
