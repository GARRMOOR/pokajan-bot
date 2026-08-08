"""A sensitivity check on the payout table, now that the real numbers are known.

The payout-to-stack ratio decides whether games end by the deck running out or by
someone going broke, and those two regimes play completely differently. The real
table is confirmed, so the 1.0x row is the game we are actually modelling — the
other rows exist to answer two questions:

  * How sensitive is the game to the payouts being slightly off? If the endings
    shift wildly between 0.8x and 1.2x, a small transcription error in one row
    would matter a lot and the table deserves re-checking.
  * Does the simulated game *length* match reality? Turn and Pokajan counts at
    1.0x should look like a real round. If they do not, something other than the
    payouts is wrong.

    python scripts/calibrate_stakes.py
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pokajan.agents.base import GreedyCallerAgent  # noqa: E402
from pokajan.core.engine import Engine  # noqa: E402
from pokajan.core.rules import Rules, load_default  # noqa: E402
from pokajan.envs.driver import play_game  # noqa: E402

GAMES = 300
SCALES = [0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0]


def variant(base: Rules, scale: float) -> Rules:
    """The live rules with every payout multiplied by `scale`.

    Scales the whole table uniformly rather than tweaking one row, so the shape of
    the real payouts — including the uneven monochrome premiums — is preserved.
    """
    table = base.raw["payouts"]["table"]
    raw = {**base.raw}
    raw["payouts"] = {
        **base.raw["payouts"],
        "table": {
            "triple": {k: max(1, int(v * scale)) for k, v in table["triple"].items()},
            "group": {
                size: {k: max(1, int(v * scale)) for k, v in row.items()}
                for size, row in table["group"].items()
            },
        },
        "bonus": {
            **base.raw["payouts"]["bonus"],
            "per_copy": int(base.bonus_per_copy * scale),
        },
    }
    return Rules.from_dict(raw, path=base.path)


def main() -> int:
    base = load_default()

    print(f"roster {base.cards.n_chars} characters, groups "
          f"{[len(m) for m in base.cards.group_members]}, "
          f"{base.play.initial_coins} coins, {GAMES} games per row")
    print("greedy callers, so these are upper bounds on how fast coins move.\n")
    print(f"{'scale':>7} {'deck_empty':>11} {'bankrupt':>9} {'turns':>7} "
          f"{'Pokajans':>9} {'minted':>8}")
    print("-" * 56)

    for scale in SCALES:
        rules = variant(base, scale)
        endings: Counter[str] = Counter()
        turns, calls, minted = [], [], []
        for i in range(GAMES):
            engine = Engine.new_game(rules, seed=i)
            result = play_game(
                engine,
                [GreedyCallerAgent(rules, seed=i * 10 + s) for s in range(rules.play.players)],
            )
            endings[result.end_reason] += 1
            turns.append(result.turns)
            calls.append(sum(result.calls_made))
            minted.append(result.coins_minted)

        marker = "  <- the real payout table" if scale == 1.0 else ""
        print(
            f"{scale:>6.2f}x "
            f"{100 * endings['deck_empty'] / GAMES:>10.0f}% "
            f"{100 * endings['bankrupt'] / GAMES:>8.0f}% "
            f"{statistics.mean(turns):>7.1f} "
            f"{statistics.mean(calls):>9.1f} "
            f"{statistics.mean(minted):>8.0f}{marker}"
        )

    print(
        "\nThe 1.0x row is the confirmed table. Compare its turns and Pokajans per\n"
        "game against a real round — if they disagree, the mismatch is somewhere\n"
        "other than the payouts (deck composition and hand limit are the suspects).\n"
        "\nIf the endings barely move between 0.8x and 1.25x, the table is not on a\n"
        "knife edge and small errors in it are survivable. If they swing hard, it is\n"
        "worth re-checking the numbers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
