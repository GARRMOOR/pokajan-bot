"""Play a batch of random games and print what actually happened.

Passing invariant tests only prove nothing is *contradictory*; they say nothing
about whether the simulated game resembles the real one. This prints the numbers
to sanity-check that against a few real rounds at M2: how long a game runs, how
often Pokajan gets called, how often each ending fires.

    python scripts/smoke.py [games] [--greedy]
"""

from __future__ import annotations

import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pokajan.agents.base import GreedyCallerAgent, RandomAgent  # noqa: E402
from pokajan.core.engine import Engine  # noqa: E402
from pokajan.core.rules import load_default  # noqa: E402
from pokajan.envs.driver import play_game  # noqa: E402


def main() -> int:
    games = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 500
    greedy = "--greedy" in sys.argv

    rules = load_default()
    print(f"rules      {rules.name}  ({rules.rules_hash[:12]})")
    print(f"roster     {rules.cards.n_chars} characters, {rules.cards.n_slots} slots")
    print(f"groups     {[len(m) for m in rules.cards.group_members]}")
    print(f"agents     {'greedy caller' if greedy else 'uniform random'}")
    print()

    results = []
    started = time.perf_counter()
    for i in range(games):
        engine = Engine.new_game(rules, seed=i)
        agents = [
            GreedyCallerAgent(rules, seed=i * 10 + s) if greedy else RandomAgent(seed=i * 10 + s)
            for s in range(rules.play.players)
        ]
        results.append(play_game(engine, agents))
    elapsed = time.perf_counter() - started

    decisions = sum(r.decisions for r in results)
    turns = [r.turns for r in results]
    calls = [sum(r.calls_made) for r in results]
    minted = [r.coins_minted for r in results]
    endings = Counter(r.end_reason for r in results)

    print(f"{games} games in {elapsed:.2f}s "
          f"({games / elapsed:,.0f} games/s, {decisions / elapsed:,.0f} decisions/s)")
    print()
    print(f"turns/game      mean {statistics.mean(turns):6.1f}   "
          f"median {statistics.median(turns):5.0f}   max {max(turns)}")
    print(f"Pokajans/game   mean {statistics.mean(calls):6.2f}   "
          f"median {statistics.median(calls):5.0f}   max {max(calls)}")
    print(f"decisions/game  mean {decisions / games:6.1f}")
    print(f"coins minted    mean {statistics.mean(minted):6.1f}   "
          f"nonzero in {sum(1 for m in minted if m):d}/{games} games")
    print()
    print("endings:")
    for reason, n in endings.most_common():
        print(f"  {reason:12} {n:5d}  ({100 * n / games:.1f}%)")
    print()

    spread = [max(r.final_coins) - min(r.final_coins) for r in results]
    print(f"final coin spread   mean {statistics.mean(spread):.0f}   max {max(spread)}")
    print(f"mean final coins    {[round(statistics.mean(r.final_coins[s] for r in results)) for s in range(4)]}")
    print("  (seats should be close under symmetric random play; a big gap means seat bias)")

    zero_call = sum(1 for r in results if sum(r.calls_made) == 0)
    if zero_call:
        print(f"\nWARNING: {zero_call}/{games} games had no Pokajan at all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
