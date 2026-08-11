"""The Pokajan engine.

Play is driven by asking whoever must decide next, rather than stepping seats in a
fixed order. A fixed order is impossible here: any player can claim a discard out
of turn, so between one seat's discard and the next seat's turn there is a window
where three other seats all have something to say.

The public surface is two methods:

    engine.pending_decisions() -> list[DecisionRequest]
    engine.submit(responses)

`pending_decisions` returns more than one request only during a claim window, and
that is the whole point of the design — see `_open_claim_window`.
"""

from __future__ import annotations

import random
import uuid

from .actions import DecisionType, call_action, legal_mask
from .evaluate import best_call, can_call, can_call_using
from .rules import Payer, Rules
from .state import EndReason, GameState, Phase, build_deck
from ..server.protocol import DecisionRequest, GameSetup, PublicState, SeatView

RECENT_DISCARDS_KEPT = 8


class Engine:
    """One game of Pokajan."""

    def __init__(self, state: GameState) -> None:
        self.state = state

    # ------------------------------------------------------------- setup ----
    def _event(self, kind: str, **data) -> None:
        """Append to the transcript, when one is being kept.

        The flag check rather than an unconditional append is deliberate: this is
        called several times per decision, and training does tens of millions of
        them without ever reading the result.
        """
        if self.state.record_events:
            self.state.events.append(
                {"turn": self.state.turn_index, "kind": kind, **data}
            )

    @classmethod
    def new_game(
        cls,
        rules: Rules,
        *,
        seed: int | None = None,
        game_id: str | None = None,
        bonus_character: int | None = None,
        record_events: bool = False,
    ) -> "Engine":
        rng = random.Random(seed)
        players = rules.play.players

        if bonus_character is None:
            bonus_character = rules.bonus_character
        if bonus_character is None:
            bonus_character = rng.randrange(rules.cards.n_chars)

        state = GameState(
            rules=rules,
            game_id=game_id or uuid.uuid4().hex[:12],
            bonus_character=bonus_character,
            deck=build_deck(rules, rng),
            hands=[rules.cards.zeros() for _ in range(players)],
            coins=[rules.play.initial_coins] * players,
            table=rules.cards.zeros(),
            scored=rules.cards.zeros(),
            discards=[rules.cards.zeros() for _ in range(players)],
            recent_discards=[[] for _ in range(players)],
            calls_made=[0] * players,
            coins_won=[0] * players,
            coins_paid=[0] * players,
            rng=rng,
            record_events=record_events,
        )
        engine = cls(state)
        engine._event("deal", bonus_character=bonus_character, deck_size=len(state.deck))
        engine._deal()
        return engine

    def _deal(self) -> None:
        s = self.state
        for _ in range(s.rules.play.deal_size):
            for seat in range(s.players):
                if not self._draw(seat):
                    return
        s.current_seat = 0
        s.turn_index = 0
        self._begin_turn(draw=True)

    def clone(self) -> "Engine":
        return Engine(self.state.clone())

    # ------------------------------------------------------- the interface --
    def pending_seats(self) -> list[tuple[int, DecisionType, int | None]]:
        """Who must act and why, as `(seat, decision, claimable_slot)`.

        The same information `pending_decisions` carries, minus the description of
        it. Building a `DecisionRequest` copies every count vector in the game so a
        client can render it, which is exactly right for a client and far too
        expensive for search: PIMC runs thousands of these per decision and never
        looks at a `PublicState`. Both methods answer from this one, so there is no
        second copy of the turn logic to drift.
        """
        s = self.state
        if s.finished:
            return []

        if s.phase is Phase.AWAIT_DISCARD:
            return [(s.current_seat, DecisionType.DISCARD, None)]

        if s.phase is Phase.AWAIT_IN_TURN_CALL:
            return [(s.current_seat, DecisionType.IN_TURN_CALL, None)]

        if s.phase is Phase.AWAIT_CHAIN:
            return [(s.chain_seat, DecisionType.CHAIN, None)]

        if s.phase is Phase.AWAIT_CLAIMS:
            return [
                (seat, DecisionType.CLAIM, s.last_discard_slot)
                for seat in s.claim_eligible
                if seat not in s.claim_responses
            ]

        raise AssertionError(f"unhandled phase {s.phase!r}")

    def pending_decisions(self) -> list[DecisionRequest]:
        """Who must act, and what they may do.

        During a claim window this returns one request per eligible seat, all
        built from the same unmodified state. Callers may answer them in any
        order, or all at once; nothing moves until every one is in.
        """
        return [
            self._request(seat, decision, claimable_slot=slot)
            for seat, decision, slot in self.pending_seats()
        ]

    def apply(self, moves) -> None:
        """Apply `(seat, action)` pairs, then run play on to the next decision.

        The same validation `submit` performs, without building a request per seat
        to validate against. This is what search drives the engine through.
        """
        s = self.state
        if s.finished:
            raise ValueError("game is finished")

        pending = {seat: (decision, slot) for seat, decision, slot in self.pending_seats()}
        for seat, action in moves:
            entry = pending.get(seat)
            if entry is None:
                raise ValueError(
                    f"seat {seat} was not asked to act (waiting on {sorted(pending)})"
                )
            decision, slot = entry
            mask = legal_mask(
                s.rules,
                decision,
                s.hands[seat],
                bonus_character=s.bonus_character,
                claimable_slot=slot,
            )
            if not (0 <= action < len(mask)):
                raise ValueError(f"action {action} out of range")
            if not mask[action]:
                raise ValueError(
                    f"seat {seat} played illegal action {action} in phase {s.phase.name}"
                )

        if s.phase is Phase.AWAIT_CLAIMS:
            for seat, action in moves:
                s.claim_responses[seat] = action
            if len(s.claim_responses) == len(s.claim_eligible):
                self._resolve_claims()
            return

        if len(moves) != 1:
            raise ValueError(f"phase {s.phase.name} expects exactly one response")
        seat, action = moves[0]

        if s.phase is Phase.AWAIT_DISCARD:
            self._apply_discard(seat, action)
        elif s.phase is Phase.AWAIT_IN_TURN_CALL:
            if action == call_action(s.rules):
                self._start_chain(seat, from_claim=False)
            else:
                s.phase = Phase.AWAIT_DISCARD
        elif s.phase is Phase.AWAIT_CHAIN:
            if action == call_action(s.rules):
                self._score_once(seat, claimed=False)
            else:
                self._end_chain()
        else:  # pragma: no cover - phases are exhaustive
            raise AssertionError(f"unhandled phase {s.phase!r}")

    def submit(self, responses) -> None:
        """Apply one or more `ActionResponse`s. The client-facing entry point."""
        if not isinstance(responses, (list, tuple)):
            responses = [responses]
        self.apply([(r.seat, r.action) for r in responses])

    # ------------------------------------------------------------- turns ----
    def _begin_turn(self, *, draw: bool) -> None:
        s = self.state
        if draw and not self._draw(s.current_seat):
            return
        if can_call(s.rules, s.hands[s.current_seat], bonus_character=s.bonus_character):
            s.phase = Phase.AWAIT_IN_TURN_CALL
        else:
            s.phase = Phase.AWAIT_DISCARD

    def _advance_turn(self) -> None:
        s = self.state
        if s.finished:
            return
        s.current_seat = (s.current_seat + 1) % s.players
        s.turn_index += 1
        self._begin_turn(draw=True)

    def _draw(self, seat: int) -> bool:
        """Move one card from the deck to a hand. False if the game just ended.

        The deck-empty ending fires when a draw is *needed* and cannot be met, so
        the last card of the deck is always played rather than stranded.
        """
        s = self.state
        if not s.deck:
            self._finish(EndReason.DECK_EMPTY)
            return False
        s.hands[seat][s.deck.pop()] += 1
        return True

    def _refill(self, seat: int, target: int) -> None:
        s = self.state
        while sum(s.hands[seat]) < target:
            if not self._draw(seat):
                return

    def _refill_target(self) -> int:
        """How many cards a call refills you to.

        Under `to_pre_discard_size` the refill restores the hand you had before
        calling: in turn you had drawn and still owed a discard, so that is
        hand_limit + 1; on a claim you were at rest, so hand_limit.

        The distinction is not cosmetic. Refilling flat to hand_limit and then
        taking the in-turn discard leaves you one card short *permanently* —
        drawing back to seven and discarding to six every turn thereafter.
        """
        s = self.state
        limit = s.rules.play.hand_limit
        if s.rules.play.refill_policy == "to_pre_discard_size" and s.chain_needs_discard:
            return limit + 1
        return limit

    def _apply_discard(self, seat: int, slot: int) -> None:
        s = self.state
        s.hands[seat][slot] -= 1
        s.table[slot] += 1
        s.discards[seat][slot] += 1
        s.recent_discards[seat].insert(0, slot)
        del s.recent_discards[seat][RECENT_DISCARDS_KEPT:]
        s.last_discard_slot = slot
        s.last_discard_seat = seat
        self._event("discard", seat=seat, slot=slot, card=s.rules.cards.describe_slot(slot))
        self._open_claim_window(seat, slot)

    # ------------------------------------------------------------ claims ----
    def _open_claim_window(self, discarder: int, slot: int) -> None:
        """Ask every seat that could use this card, all against the same state.

        The alternative — walking seats one at a time and acting on each answer —
        would let a later seat infer that an earlier one passed. That inference is
        not available in the real game, so an agent trained against it would learn
        a read that silently evaporates in live play. Nothing here mutates state
        until every eligible seat has answered.
        """
        s = self.state
        eligible = []
        for seat in self.claim_priority_order(discarder):
            probe = s.hands[seat][:]
            probe[slot] += 1
            if can_call_using(s.rules, probe, slot, bonus_character=s.bonus_character):
                eligible.append(seat)

        if not eligible:
            self._advance_turn()
            return

        s.claim_eligible = eligible
        s.claim_responses = {}
        s.phase = Phase.AWAIT_CLAIMS

    def claim_priority_order(self, discarder: int) -> list[int]:
        """Seats in the order that breaks a tie between equal payouts.

        Confirmed: ties go to "the earliest person in the play order". Whether
        that counts from the discarder or from seat zero is a config knob, since
        the two only ever differ on an exact payout tie.
        """
        s = self.state
        if s.rules.tiebreak_from == "seat_zero":
            return [seat for seat in range(s.players) if seat != discarder]
        return [(discarder + i) % s.players for i in range(1, s.players)]

    def _resolve_claims(self) -> None:
        s = self.state
        slot = s.last_discard_slot
        call_id = call_action(s.rules)

        # `claim_eligible` is already in priority order, so the first seat holding
        # the top payout wins — earliest in play order breaks the tie for free.
        winner, best_payout = None, -1
        for seat in s.claim_eligible:
            if s.claim_responses.get(seat) != call_id:
                continue
            probe = s.hands[seat][:]
            probe[slot] += 1
            # must_use: the claim is only worth what the *claimed card* completes,
            # not what the seat could already score without it.
            call = best_call(
                s.rules, probe, bonus_character=s.bonus_character, claimed=True, must_use=slot
            )
            if call is not None and call.payout > best_payout:
                winner, best_payout = seat, call.payout

        contenders = [seat for seat in s.claim_eligible if s.claim_responses.get(seat) == call_id]
        s.claim_eligible = []
        s.claim_responses = {}

        if winner is None:
            # Nobody wanted it; the card stays on the table, now unclaimable.
            self._event("claim_passed", slot=slot, card=s.rules.cards.describe_slot(slot))
            self._advance_turn()
            return

        self._event(
            "claim_won",
            seat=winner,
            slot=slot,
            card=s.rules.cards.describe_slot(slot),
            from_seat=s.last_discard_seat,
            contenders=contenders,
            payout=best_payout,
        )
        s.table[slot] -= 1
        s.hands[winner][slot] += 1
        self._start_chain(winner, from_claim=True, claimed_slot=slot)

    # ------------------------------------------------- calls and chaining ---
    def _start_chain(
        self, seat: int, *, from_claim: bool, claimed_slot: int | None = None
    ) -> None:
        s = self.state
        s.chain_seat = seat
        s.chain_from_claim = from_claim
        s.chain_needs_discard = (
            s.rules.play.discard_after_claim
            if from_claim
            else s.rules.play.discard_after_in_turn_call
        )
        self._score_once(seat, claimed=from_claim, must_use=claimed_slot)

    def _score_once(self, seat: int, *, claimed: bool, must_use: int | None = None) -> None:
        """Score the best hand once, pay for it, refill, and offer another call.

        `must_use` is set only for the first call of a claim: that one has to
        spend the claimed card. Everything later in the chain is scored from the
        hand freely, since those cards were drawn, not taken.
        """
        s = self.state
        call = best_call(
            s.rules,
            s.hands[seat],
            bonus_character=s.bonus_character,
            claimed=claimed,
            must_use=must_use,
        )
        if call is None:
            self._end_chain()
            return

        for slot, spend in enumerate(call.cards):
            if spend:
                s.hands[seat][slot] -= spend
                s.scored[slot] += spend
        s.calls_made[seat] += 1
        self._event(
            "pokajan",
            seat=seat,
            claimed=claimed,
            payout=call.payout,
            hand_kind=call.kind.value,
            monochrome=call.monochrome,
            bonus_copies=call.bonus_copies,
            group_size=None if call.group is None else len(s.rules.cards.group_members[call.group]),
            label=call.describe(s.rules.cards),
            cards=[
                s.rules.cards.describe_slot(i)
                for i, n in enumerate(call.cards)
                for _ in range(n)
            ],
        )

        self._pay(seat, call.payout, claimed=claimed)
        if s.finished:
            return

        # Every refill drains the shared deck, which is why chaining is a choice:
        # a long chain can run the deck dry and end the game.
        before = sum(s.hands[seat])
        target = self._refill_target()
        self._refill(seat, target)
        self._event("refill", seat=seat, drawn=sum(s.hands[seat]) - before,
                    to=target, deck_remaining=len(s.deck))
        if s.finished:
            return

        if s.rules.play.chain_allowed and can_call(
            s.rules, s.hands[seat], bonus_character=s.bonus_character
        ):
            s.phase = Phase.AWAIT_CHAIN
        else:
            self._end_chain()

    def _end_chain(self) -> None:
        s = self.state
        seat = s.chain_seat
        needs_discard = s.chain_needs_discard
        from_claim = s.chain_from_claim
        s.chain_seat = None
        s.chain_needs_discard = False
        s.chain_from_claim = False

        if needs_discard:
            s.current_seat = seat
            s.phase = Phase.AWAIT_DISCARD
            return

        if from_claim and s.rules.play.claim_moves_turn:
            s.current_seat = seat
        self._advance_turn()

    # ------------------------------------------------------------ payment ---
    def _pay(self, winner: int, amount: int, *, claimed: bool) -> None:
        """Move coins, applying the asymmetric bankruptcy floor.

        The caller always gains the full amount. A payer who cannot cover their
        share stops at zero and the difference is minted — so coins are not
        conserved, and `coins_minted` records exactly how much was created.
        """
        s = self.state
        if s.rules.payer_for(claimed) is Payer.DISCARDER:
            payers = [s.last_discard_seat]
        else:
            payers = [(winner + i) % s.players for i in range(1, s.players)]

        share, remainder = divmod(amount, len(payers))
        # CONFIRMED that no real payout is indivisible by three, so `remainder` is
        # always zero in practice — see test_every_real_payout_splits_evenly. The
        # branch stays for generated configs in the property tests, where it keeps
        # the totals exact by giving the odd coins to the earliest payers.
        owed = [share + (1 if i < remainder else 0) for i in range(len(payers))]

        collected = 0
        breakdown = []
        for payer, due in zip(payers, owed):
            paid = min(due, s.coins[payer])
            s.coins[payer] -= paid
            s.coins_paid[payer] += paid
            collected += paid
            breakdown.append({"seat": payer, "owed": due, "paid": paid})

        s.coins[winner] += amount
        s.coins_won[winner] += amount
        minted = amount - collected
        s.coins_minted += minted

        self._event(
            "payment",
            to_seat=winner,
            amount=amount,
            payers=breakdown,
            minted=minted,
            coins=s.coins[:],
        )

        floor = s.rules.end.coin_floor
        if any(c <= floor for c in s.coins):
            self._finish(EndReason.BANKRUPT)

    def _finish(self, reason: EndReason) -> None:
        s = self.state
        self._event(
            "game_over",
            reason=reason.value,
            coins=s.coins[:],
            standings=sorted(range(s.players), key=lambda seat: (-s.coins[seat], seat)),
            minted=s.coins_minted,
        )
        s.finished = True
        s.end_reason = reason.value
        s.phase = Phase.FINISHED
        s.chain_seat = None
        s.claim_eligible = []
        s.claim_responses = {}

    # ------------------------------------------------------------- views ----
    def setup(self) -> GameSetup:
        s = self.state
        space = s.rules.cards
        return GameSetup(
            protocol_version=1,
            rules_hash=s.rules.rules_hash,
            rules_name=s.rules.name,
            game_id=s.game_id,
            character_ids=list(space.character_ids),
            character_names=list(space.character_names),
            colors=list(space.colors),
            group_ids=list(space.group_ids),
            group_names=list(space.group_names),
            group_members=[list(m) for m in space.group_members],
            n_slots=space.n_slots,
            hand_limit=s.rules.play.hand_limit,
            players=s.players,
            initial_coins=s.rules.play.initial_coins,
            deck_size=s.rules.deck_size,
            bonus_character=s.bonus_character,
        )

    def public_state(self, seat: int) -> PublicState:
        """One seat's legitimate view. Never includes another hand or deck order."""
        s = self.state
        return PublicState(
            game_id=s.game_id,
            turn_index=s.turn_index,
            viewer=seat,
            hand=s.hands[seat][:],
            seats=[
                SeatView(
                    seat=other,
                    coins=s.coins[other],
                    hand_size=s.hand_size(other),
                    discards=s.discards[other][:],
                    calls_made=s.calls_made[other],
                    coins_won=s.coins_won[other],
                    coins_paid=s.coins_paid[other],
                )
                for other in range(s.players)
            ],
            deck_remaining=s.deck_remaining,
            table=s.table[:],
            scored=s.scored[:],
            last_discard_slot=s.last_discard_slot,
            last_discard_seat=s.last_discard_seat,
            current_seat=s.current_seat,
            phase=s.phase.name,
            finished=s.finished,
            final_coins=s.coins[:] if s.finished else None,
            coins_minted=s.coins_minted,
            bonus_character=s.bonus_character,
            recent_discards=[r[:] for r in s.recent_discards],
        )

    def _request(
        self, seat: int, decision: DecisionType, *, claimable_slot: int | None = None
    ) -> DecisionRequest:
        s = self.state
        mask = legal_mask(
            s.rules,
            decision,
            s.hands[seat],
            bonus_character=s.bonus_character,
            claimable_slot=claimable_slot,
        )

        payout, label = None, None
        if mask[call_action(s.rules)]:
            probe = s.hands[seat]
            if claimable_slot is not None:
                probe = probe[:]
                probe[claimable_slot] += 1
            call = best_call(
                s.rules,
                probe,
                bonus_character=s.bonus_character,
                claimed=claimable_slot is not None,
                must_use=claimable_slot,
            )
            if call is not None:
                payout, label = call.payout, call.describe(s.rules.cards)

        return DecisionRequest(
            game_id=s.game_id,
            seat=seat,
            decision=decision.name,
            legal_mask=mask,
            state=self.public_state(seat),
            claimable_slot=claimable_slot,
            best_call_payout=payout,
            best_call_label=label,
        )

    # ------------------------------------------------------------ results ---
    @property
    def finished(self) -> bool:
        return self.state.finished

    def final_coins(self) -> list[int]:
        return self.state.coins[:]

    def standings(self) -> list[int]:
        """Seats ordered best to worst by final coins, ties by seat order."""
        s = self.state
        return sorted(range(s.players), key=lambda seat: (-s.coins[seat], seat))
