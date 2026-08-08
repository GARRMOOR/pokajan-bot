"""Loading and interpreting rules/pokajan_v1.yaml.

Every gameplay number the engine uses arrives through this module. Nothing under
pokajan/core/ may define a rule constant of its own — when the real payout table
turns out to differ from our guesses, the fix has to be a YAML edit, not a code
change. The engine reads `Rules`; `Rules` reads the file.

The `rules_hash` exists because a trained policy is only valid for the rules it
was trained under. Silently loading a checkpoint against edited rules would give
plausible-looking but meaningless play, so the hash is stamped into every
checkpoint and eval report and checked on load.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from .cards import CardSpace


class HandKind(str, Enum):
    TRIPLE = "triple"
    GROUP = "group"


class Payer(str, Enum):
    """Who hands over the coins for a scored hand."""

    DISCARDER = "discarder"          # the claimed player pays the whole amount alone
    SPLIT_OTHERS = "split_others"    # the other three split it evenly


class Composition(str, Enum):
    EVEN_AS_POSSIBLE = "even_as_possible"
    UNIFORM_RANDOM = "uniform_random"


@dataclass(frozen=True)
class PlayRules:
    players: int
    initial_coins: int
    hand_limit: int
    deal_size: int
    draws_per_turn: int
    refill_policy: str
    chain_allowed: bool
    chain_is_optional: bool
    claim_moves_turn: bool
    discard_after_claim: bool
    discard_after_in_turn_call: bool
    claimable: str


@dataclass(frozen=True)
class EndConditions:
    deck_empty: bool
    coin_floor: int
    floor_is_asymmetric: bool
    multi_round: bool


@dataclass(frozen=True)
class Rules:
    """Frozen, validated view of one rules file."""

    version: int
    name: str
    path: Path
    rules_hash: str
    raw: dict[str, Any]

    cards: CardSpace
    play: PlayRules
    end: EndConditions

    deck_size: int
    composition: Composition
    bonus_character: int | None      # index into cards.character_ids; None = random per game
    tiebreak_order: tuple[str, ...]
    # Whether "earliest in play order" counts from the discarder or from seat 0.
    # The two only ever differ on an exact payout tie. Optional, so the frozen test
    # fixture stays untouched.
    tiebreak_from: str

    # payout table, pre-resolved into plain numbers at load time
    _base_triple: float
    _base_group: dict[int, float]
    _mod_monochrome: tuple[str, float]
    _mod_bonus: tuple[str, float]
    _mod_claimed: tuple[str, float]
    _combine: str
    _rounding: str
    _payer_when_claimed: Payer
    _payer_otherwise: Payer
    caller_gains_full_amount_on_payer_bankruptcy: bool

    # -------------------------------------------------------------- load ----
    @classmethod
    def load(cls, path: str | Path) -> "Rules":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls.from_dict(raw, path=path)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], path: str | Path = "<memory>") -> "Rules":
        deck = raw["deck"]
        play = raw["play"]
        pay = raw["payouts"]
        end = raw["end_conditions"]

        cards = CardSpace.from_config(
            characters=raw["characters"],
            groups=raw["groups"],
            colors=deck["colors"],
            max_per_color=deck["max_per_color"],
        )

        bonus = raw.get("bonus_character")
        bonus_idx = cards.char_index(bonus) if bonus is not None else None

        base_group = {int(k): float(v) for k, v in pay["base"]["group"].items()}
        mods = pay["modifiers"]

        rules = cls(
            version=int(raw["version"]),
            name=str(raw["name"]),
            path=Path(path),
            rules_hash=canonical_hash(raw),
            raw=raw,
            cards=cards,
            play=PlayRules(
                players=int(play["players"]),
                initial_coins=int(play["initial_coins"]),
                hand_limit=int(play["hand_limit"]),
                deal_size=int(play["deal_size"]),
                draws_per_turn=int(play["draws_per_turn"]),
                refill_policy=str(play["refill_policy"]),
                chain_allowed=bool(play["chain_allowed"]),
                chain_is_optional=bool(play["chain_is_optional"]),
                claim_moves_turn=bool(play["claim_moves_turn"]),
                discard_after_claim=bool(play["discard_after_claim"]),
                discard_after_in_turn_call=bool(play["discard_after_in_turn_call"]),
                claimable=str(play["claimable"]),
            ),
            end=EndConditions(
                deck_empty=bool(end["deck_empty"]),
                coin_floor=int(end["coin_floor"]),
                floor_is_asymmetric=bool(end["floor_is_asymmetric"]),
                multi_round=bool(end["multi_round"]),
            ),
            deck_size=int(deck["size"]),
            composition=Composition(deck["composition"]),
            bonus_character=bonus_idx,
            tiebreak_order=tuple(raw["tiebreak"]["order"]),
            tiebreak_from=str(raw["tiebreak"].get("turn_order_from", "discarder")),
            _base_triple=float(pay["base"]["triple"]),
            _base_group=base_group,
            _mod_monochrome=(mods["monochrome"]["mode"], float(mods["monochrome"]["value"])),
            _mod_bonus=(mods["bonus_card"]["mode"], float(mods["bonus_card"]["value"])),
            _mod_claimed=(mods["claimed"]["mode"], float(mods["claimed"]["value"])),
            _combine=str(pay["combine"]),
            _rounding=str(pay["rounding"]),
            _payer_when_claimed=Payer(pay["payer"]["when_claimed"]),
            _payer_otherwise=Payer(pay["payer"]["otherwise"]),
            caller_gains_full_amount_on_payer_bankruptcy=bool(
                pay["caller_gains_full_amount_on_payer_bankruptcy"]
            ),
        )
        rules.validate()
        return rules

    # ---------------------------------------------------------- validate ----
    def validate(self) -> None:
        """Reject configs the engine could not play, with an actionable message.

        This runs on every load, including the randomly generated configs the
        property tests build, so it doubles as the definition of "a config the
        engine promises to handle".
        """
        c = self.cards
        if self.play.players < 2:
            raise ValueError("play.players must be at least 2")
        if c.n_chars < 1:
            raise ValueError("roster is empty")
        if c.n_colors < 1:
            raise ValueError("deck.colors is empty")
        if c.max_per_color < 1:
            raise ValueError("deck.max_per_color must be >= 1")

        # The deck must be buildable: 100 cards cannot be drawn from a roster that
        # cannot hold 100 cards. This is the check that catches a roster edit that
        # dropped too many characters.
        capacity = c.n_slots * c.max_per_color
        if self.deck_size > capacity:
            raise ValueError(
                f"deck.size {self.deck_size} exceeds capacity {capacity} "
                f"({c.n_chars} characters x {c.n_colors} colours x {c.max_per_color} copies)"
            )

        # ...and it must survive the deal.
        needed = self.play.deal_size * self.play.players
        if needed > self.deck_size:
            raise ValueError(
                f"dealing {self.play.deal_size} to {self.play.players} players needs "
                f"{needed} cards but the deck holds {self.deck_size}"
            )

        if self.play.hand_limit < self.play.deal_size:
            raise ValueError("play.hand_limit must be >= play.deal_size")

        # Every group must be completable, or a whole hand type is dead.
        for gi, members in enumerate(c.group_members):
            if not members:
                raise ValueError(f"group {c.group_ids[gi]!r} has no members")
            if len(members) > self.play.hand_limit:
                raise ValueError(
                    f"group {c.group_ids[gi]!r} has {len(members)} members but the hand "
                    f"limit is {self.play.hand_limit} — it could never be completed"
                )
            if len(members) not in self._base_group:
                raise ValueError(
                    f"group {c.group_ids[gi]!r} has size {len(members)} but "
                    f"payouts.base.group has no entry for that size "
                    f"(has {sorted(self._base_group)})"
                )

        ungrouped = [
            c.character_ids[i] for i in range(c.n_chars) if not c.char_groups[i]
        ]
        if ungrouped:
            raise ValueError(f"characters belong to no group: {ungrouped}")

        if self._combine not in ("multiplicative", "additive"):
            raise ValueError(f"unknown payouts.combine {self._combine!r}")
        if self._rounding not in ("nearest_1", "nearest_10", "floor", "ceil"):
            raise ValueError(f"unknown payouts.rounding {self._rounding!r}")
        if self.tiebreak_from not in ("discarder", "seat_zero"):
            raise ValueError(f"unknown tiebreak.turn_order_from {self.tiebreak_from!r}")

    # ------------------------------------------------------------ payout ----
    def payout(
        self,
        kind: HandKind,
        *,
        group_size: int | None = None,
        monochrome: bool = False,
        bonus: bool = False,
        claimed: bool = False,
    ) -> int:
        """Coins the caller gains for one scored hand.

        This is also the hand's *strength*: the confirmed tiebreak is by payout,
        so there is no separate ranking notion anywhere in the codebase.
        """
        if kind is HandKind.TRIPLE:
            amount = self._base_triple
        else:
            if group_size is None:
                raise ValueError("group_size is required for group hands")
            amount = self._base_group[group_size]

        for active, (mode, value) in (
            (monochrome, self._mod_monochrome),
            (bonus, self._mod_bonus),
            (claimed, self._mod_claimed),
        ):
            if not active:
                continue
            if mode == "multiply":
                amount = amount * value if self._combine == "multiplicative" else amount + (amount * (value - 1.0))
            elif mode == "add":
                amount += value
            else:
                raise ValueError(f"unknown modifier mode {mode!r}")

        return self._round(amount)

    def _round(self, amount: float) -> int:
        if self._rounding == "nearest_1":
            return int(amount + 0.5)
        if self._rounding == "nearest_10":
            return int(round(amount / 10.0)) * 10
        if self._rounding == "floor":
            return int(amount)
        return -int(-amount // 1)  # ceil

    def payer_for(self, claimed: bool) -> Payer:
        return self._payer_when_claimed if claimed else self._payer_otherwise


def canonical_hash(raw: dict[str, Any]) -> str:
    """Stable sha256 of a rules dict.

    Canonicalised through JSON with sorted keys so that comment edits, key
    reordering and whitespace do not invalidate checkpoints, while any change to
    an actual value does.
    """
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "rules" / "pokajan_v1.yaml"


def load_default() -> Rules:
    return Rules.load(DEFAULT_RULES_PATH)
