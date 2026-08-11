"""Measuring whether one agent is actually better than another.

Pokajan is extremely high variance. A single game swings by well over a thousand
coins on one monochrome five-group, so a naive "play 200 games and compare means"
will confidently report an edge that does not exist. Two things fix that, and both
are structural rather than a matter of playing more games.

**Duplicate dealing.** The deck is fixed per seed, and the same deal is replayed
once per seat with the test agent in each chair. Every point of seat advantage and
every lucky opening hand is then experienced by both sides, so it cancels instead
of accumulating as noise.

**Seeds are the sample, not games.** The four rotations of one seed share a deck
and are therefore correlated; treating them as four independent observations
understates the error by roughly half and is the single easiest way to fool
yourself here. Error bars below are computed over per-seed means.

The headline numbers are chosen to match the two agents the project is aiming at:
mean coin delta for the max-EV agent, `P(final > 1000)` for the safe one. Coins
minted per game is reported alongside because a nonzero, agent-dependent value
there is the tell that short-stack sniping is being exploited on purpose.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..agents.base import GreedyCallerAgent, RandomAgent
from ..agents.heuristic import HeuristicAgent
from ..agents.pimc import PIMCAgent
from ..agents.rollout import FastAgent
from ..core.engine import Engine
from ..core.rules import Rules, load_default
from ..envs.driver import play_game

# Agents are registered by name rather than passed as callables because the
# multiprocess path has to ship them to a worker, and on Windows that means
# spawning a fresh interpreter: a name pickles, a closure does not.
AGENTS = {
    "random": lambda rules, seed: RandomAgent(seed=seed),
    "greedy": lambda rules, seed: GreedyCallerAgent(rules, seed=seed),
    "heuristic": lambda rules, seed: HeuristicAgent(rules, seed=seed),
    # Same valuation, defence switched off. The gap between this and `heuristic`
    # is what the belief model is worth in coins.
    "heuristic-blind": lambda rules, seed: HeuristicAgent(
        rules, seed=seed, defend=False, name="heuristic-blind"
    ),
    # Fewer particles: the configuration cheap enough for PIMC rollouts at M4.
    "heuristic-fast": lambda rules, seed: HeuristicAgent(
        rules, seed=seed, particles=12, name="heuristic-fast"
    ),
    # Values a hand by combining its best four targets rather than taking the best
    # one. Measured at -38 coins/game and left in so the comparison can be re-run
    # against a better completion model; see TARGETS_COMBINED.
    "heuristic-combined": lambda rules, seed: HeuristicAgent(
        rules, seed=seed, targets_combined=4, name="heuristic-combined"
    ),
    # Determinized search, ~250ms a decision. Measured *worse* than the heuristic
    # it is built on; the variants below are the sweep that established why, kept
    # so the finding can be rechecked rather than taken on trust. See the README.
    "pimc": lambda rules, seed: PIMCAgent(rules, seed=seed),
    # Cheap valuation plus belief-driven defence. Built for M5's training loop,
    # where the opponent pool is queried far too often to afford the heuristic.
    "fast": lambda rules, seed: FastAgent(rules, seed=seed),
    # Control for the PIMC matchups: PIMC's internal prior runs on as many belief
    # particles as it has determinizations, so comparing it against the 48-particle
    # default heuristic would charge the search for a weaker prior.
    "heuristic-p16": lambda rules, seed: HeuristicAgent(
        rules, seed=seed, particles=16, name="heuristic-p16"
    ),
    # Spending a large compute budget on belief precision rather than on search
    # depth: same algorithm, a much sharper estimate of discard danger.
    "heuristic-p1024": lambda rules, seed: HeuristicAgent(
        rules, seed=seed, particles=1024, name="heuristic-p1024"
    ),
    # Sampled valuation: replaces the closed-form completion probability with
    # E[best payout reachable] over futures drawn from the belief. Draws level with
    # the closed form rather than beating it; the horizon sweep behind that is
    # recorded on DEFAULT_HORIZON.
    "heuristic-mc": lambda rules, seed: HeuristicAgent(
        rules, seed=seed, futures=32, name="heuristic-mc"
    ),
    # Truncated rollouts, to test whether the chaos of a full game was the problem.
    # It helps (-144 -> -85) and does not rescue it.
    "pimc-d12": lambda rules, seed: PIMCAgent(
        rules, seed=seed, rollout_depth=12, name="pimc-d12"
    ),
    # Search only where it is cheap and plausible — calls, claims, chains — and
    # leave discards to the heuristic. The best PIMC variant, and still level at
    # best with the agent it wraps.
    "pimc-calls": lambda rules, seed: PIMCAgent(
        rules, seed=seed, search_discards=False, name="pimc-calls"
    ),
}


def build_agent(name: str, rules: Rules, seed: int):
    try:
        return AGENTS[name](rules, seed)
    except KeyError:
        raise KeyError(f"unknown agent {name!r} (have {sorted(AGENTS)})") from None


@dataclass
class MatchupReport:
    agent: str
    baseline: str
    rules_hash: str
    seeds: int
    games: int
    mean_delta: float
    stderr: float
    win_rate: float
    p_above_start: float
    bust_rate: float
    mean_by_seat: list[float]
    coins_minted: float
    end_reasons: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        return (self.mean_delta - 1.96 * self.stderr, self.mean_delta + 1.96 * self.stderr)

    @property
    def significant(self) -> bool:
        """Whether the confidence interval excludes zero."""
        low, high = self.ci95
        return low > 0.0 or high < 0.0

    def render(self) -> str:
        low, high = self.ci95
        verdict = "significant" if self.significant else "NOT significant"
        spread = max(self.mean_by_seat) - min(self.mean_by_seat)
        lines = [
            f"{self.agent} vs {self.baseline}",
            f"  rules       {self.rules_hash[:12]}",
            f"  games       {self.games} ({self.seeds} seeds x {len(self.mean_by_seat)} seats)",
            f"  coin delta  {self.mean_delta:+.1f}  95% CI [{low:+.1f}, {high:+.1f}]  {verdict}",
            f"  win rate    {self.win_rate:.1%}",
            f"  P(>start)   {self.p_above_start:.1%}",
            f"  bust rate   {self.bust_rate:.1%}",
            f"  by seat     {[round(v) for v in self.mean_by_seat]}  (spread {spread:.0f})",
            f"  minted/game {self.coins_minted:.1f}",
            f"  endings     {self.end_reasons}",
            f"  took        {self.seconds:.1f}s",
        ]
        return "\n".join(lines)


def play_rotation(rules: Rules, agent: str, baseline: str, seed: int, rotation: int):
    """One deal, with the test agent seated at `rotation`.

    The engine seed fixes the shuffle and the bonus holomem, so every rotation of a
    given seed starts from the same deck. Play diverges immediately after that —
    different discards draw different cards — which is inherent to the game and not
    something duplicate dealing claims to remove.
    """
    engine = Engine.new_game(rules, seed=seed)
    players = rules.play.players
    seats = [
        build_agent(agent if s == rotation else baseline, rules, seed * players + s)
        for s in range(players)
    ]
    result = play_game(engine, seats)
    return {
        "seed": seed,
        "rotation": rotation,
        "final": result.final_coins[rotation],
        "won": result.standings[0] == rotation,
        "end_reason": result.end_reason,
        "minted": result.coins_minted,
    }


def _seed_task(args):
    rules = _WORKER_RULES if _WORKER_RULES is not None else load_default()
    agent, baseline, seed = args
    return [
        play_rotation(rules, agent, baseline, seed, r) for r in range(rules.play.players)
    ]


_WORKER_RULES: Rules | None = None


def _init_worker(rules_path: str) -> None:
    global _WORKER_RULES
    _WORKER_RULES = Rules.load(rules_path)


def duplicate_match(
    rules: Rules,
    agent: str,
    baseline: str,
    *,
    seeds: int = 500,
    start_seed: int = 0,
    workers: int = 1,
    progress=None,
) -> MatchupReport:
    """Play `seeds` deals, each once per seat, and report with honest error bars."""
    started = time.perf_counter()
    players = rules.play.players
    tasks = [(agent, baseline, start_seed + i) for i in range(seeds)]

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(str(rules.path),)
        ) as pool:
            batches = []
            for i, batch in enumerate(pool.map(_seed_task, tasks, chunksize=4)):
                batches.append(batch)
                if progress:
                    progress(i + 1, seeds)
    else:
        global _WORKER_RULES
        _WORKER_RULES = rules
        batches = []
        for i, task in enumerate(tasks):
            batches.append(_seed_task(task))
            if progress:
                progress(i + 1, seeds)

    initial = rules.play.initial_coins
    floor = rules.end.coin_floor
    games = [row for batch in batches for row in batch]

    # One observation per seed: the mean across its four rotations. This is the
    # step that makes the confidence interval mean what it says.
    per_seed = [
        statistics.fmean([row["final"] - initial for row in batch]) for batch in batches
    ]
    stdev = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0

    by_seat = []
    for rotation in range(players):
        rows = [row["final"] - initial for row in games if row["rotation"] == rotation]
        by_seat.append(statistics.fmean(rows) if rows else 0.0)

    end_reasons: dict[str, int] = {}
    for row in games:
        key = row["end_reason"] or "unknown"
        end_reasons[key] = end_reasons.get(key, 0) + 1

    return MatchupReport(
        agent=agent,
        baseline=baseline,
        rules_hash=rules.rules_hash,
        seeds=len(batches),
        games=len(games),
        mean_delta=statistics.fmean(per_seed) if per_seed else 0.0,
        stderr=stdev / (len(per_seed) ** 0.5) if per_seed else 0.0,
        win_rate=sum(1 for row in games if row["won"]) / len(games),
        p_above_start=sum(1 for row in games if row["final"] > initial) / len(games),
        bust_rate=sum(1 for row in games if row["final"] <= floor) / len(games),
        mean_by_seat=by_seat,
        coins_minted=statistics.fmean([row["minted"] for row in games]),
        end_reasons=end_reasons,
        seconds=time.perf_counter() - started,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", default="heuristic", choices=sorted(AGENTS))
    parser.add_argument("--baseline", default="greedy", choices=sorted(AGENTS))
    parser.add_argument(
        "--seeds",
        type=int,
        default=500,
        help="deals to play; each is replayed once per seat (default 500 = 2000 games)",
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="processes; 1 runs in-process",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    rules = Rules.load(args.rules) if args.rules else load_default()

    def progress(done: int, total: int) -> None:
        if not args.quiet and (done % 25 == 0 or done == total):
            print(f"  {done}/{total} seeds", end="\r", flush=True)

    report = duplicate_match(
        rules,
        args.agent,
        args.baseline,
        seeds=args.seeds,
        start_seed=args.start_seed,
        workers=args.workers,
        progress=None if args.quiet else progress,
    )
    if not args.quiet:
        print(" " * 40, end="\r")
    print(report.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
