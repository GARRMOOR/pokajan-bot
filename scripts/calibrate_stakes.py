"""Which ending does the payout scale produce?

The single most useful calibration question for M0b, and it needs no payout numbers
at all — just an answer to "how do your real games actually end?"

Payouts are unknown, but they are not free parameters: the ratio of payout to the
1000-coin starting stack decides whether games end by the deck running out or by
someone going broke. Those two regimes play completely differently, so getting the
scale wrong would train an agent for the wrong game even if every other rule were
right.

So: run a few real games, note whether they end on deck exhaustion or on a player
hitting zero, and read the payout scale off this table.

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
SCALES = [30, 60, 120, 200, 350, 500, 800, 1200, 2000]


def variant(base: Rules, triple_payout: int) -> Rules:
    """The live rules with the payout scale swapped out."""
    raw = {**base.raw}
    raw["payouts"] = {
        **base.raw["payouts"],
        "base": {
            "triple": triple_payout,
            "group": {
                size: max(1, triple_payout * size // 3)
                for size in base.raw["payouts"]["base"]["group"]
            },
        },
    }
    return Rules.from_dict(raw, path=base.path)


def main() -> int:
    base = load_default()
    current = base.raw["payouts"]["base"]["triple"]

    print(f"roster {base.cards.n_chars} characters, groups "
          f"{[len(m) for m in base.cards.group_members]}, "
          f"{base.play.initial_coins} coins, {GAMES} games per row")
    print("greedy callers, so these are upper bounds on how fast coins move.\n")
    print(f"{'triple':>7} {'deck_empty':>11} {'bankrupt':>9} {'turns':>7} "
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

        marker = "  <- current" if scale == current else ""
        print(
            f"{scale:>7} "
            f"{100 * endings['deck_empty'] / GAMES:>10.0f}% "
            f"{100 * endings['bankrupt'] / GAMES:>8.0f}% "
            f"{statistics.mean(turns):>7.1f} "
            f"{statistics.mean(calls):>9.1f} "
            f"{statistics.mean(minted):>8.0f}{marker}"
        )

    print(
        "\nRead it backwards from real games:\n"
        "  every game ends with the deck running out  -> payouts are low, near the top\n"
        "  endings are a mix of both                  -> the middle rows\n"
        "  someone almost always goes broke first     -> payouts are high, near the bottom\n"
        "\nTurns and Pokajans per game are a second, independent check on the same thing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
