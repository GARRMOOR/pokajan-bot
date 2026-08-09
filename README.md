# pokajan-bot

A learning agent for **Pokajan**, the four-player card minigame in *Hololive Dreams*.

Two agents are the goal: one that maximises final coins, and a safer one that
maximises the probability of finishing above the 1000-coin starting stack. Getting
there means first building a playable copy of the game to train against, then
reading the real game's screen so the bot can advise during live play.

## Status

**M7 — the bot explains itself, in the browser and in an overlay.** Full game logic,
an AEC environment, the confirmed payout table, a web table you can sit down and play
*with a hint panel*, an overlay window pinned over the screen, a two-level belief,
three agents, a determinized-search implementation, and an evaluation harness, with
148 tests passing.

What remains before this is useful against the real game is M8 alone: something that
reads `PublicState` off the screen instead of out of the simulator. The overlay is
already fed through a one-way advice socket, so that swap does not touch it.

The agent ladder, all measured by duplicate dealing with 4-seat rotation:

| agent | vs greedy | vs heuristic | cost/decision |
|---|---:|---:|---:|
| **heuristic-p1024** — 1024 belief particles | — | **+50** [+16, +84] | 248 ms |
| **heuristic** — belief-driven, expected coins | **+638** [+612, +664] | — | 24 ms |
| **fast** — cheap valuation, belief defence | +500 [+457, +544] | −104 [−144, −64] | 6 ms |
| **pimc** — determinized search | — | **−144** [−197, −92] | 250 ms |
| greedy | — | −638 | 0.02 ms |

Two results, and the contrast between them is the milestone. Determinized search —
the whole point of M4 — is *worse* than the heuristic it wraps. Spending a
comparable budget on belief precision instead is better. For almost identical
compute, one lever is worth −144 and the other +50.

`heuristic` stays the default at 48 particles rather than adopting the stronger
setting, because a tenfold cost increase to buy 8% of its edge over greedy is a bad
trade for the thing every matchup is measured against. `heuristic-p1024` is there
for hint mode, where a quarter-second is free.

Two other numbers worth keeping in view:

| | |
|---|---|
| what discard safety is worth | **+173 coins/game** against an identical agent with defence off |
| composition estimate vs the in-game counter | **9.3× more accurate** |

That last row is the project's whole thesis in one number, and it is explained
under [the one thing still assumed](#the-one-thing-still-assumed).

```powershell
.\.venv\Scripts\python -m pokajan.server.app     # then open http://127.0.0.1:8000
```

You take a seat, simple bots take the other three. The point is not the game — it
is the panels around it: every payout is shown with the arithmetic that produced it,
the transcript reads like something you can hold next to a real round and check line
by line, and the *Still guessing* tab lists the rules we are assuming rather than
knowing. Play a real round alongside it and those go away.

Press **h** and the bot says what it would do and why:

```
[DISCARD] discard Sakura Miko (pink)                 48 samples · 18 ms
  50% chance this ranking survives redrawing the belief — not a claim about the model
  Throw Sakura Miko (pink). What is left is worth about 256 coins, and it risks 18
  if somebody claims it. This one is a coin flip: Hoshimachi Suisei (pink) is
  within 0 coins, which is inside the sampling noise. Either is fine.
  next best — Hoshimachi Suisei (pink) level · Ninomae Ina'nis (blue) -2
```

Three things about that output are deliberate, and each is a way the panel could
have been worse:

- **It reports a toss-up as a toss-up.** Opening discards usually are one: six draws
  at 1024 particles on the same position produced three different answers. A panel
  that manufactured a reason each time would be persuasive and wrong.
- **The percentage is labelled with what it measures** — whether the *ranking*
  survives resampling the belief. Model error is not in it. Calling it "confidence"
  would imply a claim the number cannot support.
- **Alternatives are shown as a gap, not a score.** A score nets danger off hand
  value, so printing it beside the headline's hand value invites subtracting two
  different quantities — which is exactly what happened while building this, reading
  a 5-coin gap where the real one was nil.

The sample count is selectable, with wall time shown, because that is the decision
the overlay depends on: 48 samples costs ~16 ms, 1024 costs ~210 ms and is worth
+50 coins/game. Against the real game's ~10 s per turn, both are free — so the
overlay can afford the expensive setting, and now that is measured rather than hoped.

```
M0  core model, protocol, tests            <- done
M1  engine + environment                   <- done
M2  web GUI, human-playable                <- done
M0b capture real payouts + card art        <- payouts confirmed; card art still open
M3  observation encoder, belief, heuristic agent, eval harness   <- done
M4  PIMC agent                             <- built and measured; does not beat M3
M5  vectorised env, behaviour cloning, PPO self-play
M6  risk-conditioned training
M7  hint mode + overlay                    <- done
M8  screen reading                         <- the only thing left before live use
```

**M7 and M8 do not depend on M5 or M6**, and the roadmap's ordering is misleading
about that. It is the order for building the *best* agent, not for getting a usable
overlay: the heuristic already plays well, already decides from a `PublicState`, and
because `server/protocol.py` is the seam, dropping a trained agent in later changes
nothing else. Taking the milestones in order would be two of them of delay for no
benefit to the thing being built. See `CLAUDE.md` for the order that works.

### Measuring an agent

```powershell
.\.venv\Scripts\python -m pokajan.train.evaluate --agent heuristic --baseline greedy
```

The game is violently high-variance — one monochrome five-group swings a game by
more than the starting stack — so a plain "play 200 games and compare means" will
report edges that are not there. Two things fix that, and neither is playing more
games. The deck is **fixed per seed and replayed once per seat**, so seat
advantage and opening-hand luck are experienced by both sides and cancel. And the
four rotations of one deal are *correlated*, so the sample size is the number of
seeds, not the number of games; error bars are computed over per-seed means.
Treating 2000 games as 2000 observations understates the error by about half and
is the easiest way to fool yourself here.

The `by seat` line is the check that this worked: with duplicate dealing the
spread across seats is ~58 coins, against per-game swings in the thousands.

Registered agents: `random`, `greedy`, `heuristic`, `fast`, `heuristic-p1024` and
`pimc`, plus the ablations that produced the numbers below — `heuristic-blind`
(defence off), `heuristic-p16` (the prior PIMC actually runs on),
`heuristic-combined`, `heuristic-mc` (sampled valuation), `pimc-d12` (truncated
rollouts) and `pimc-calls` (search only call decisions). Every negative result in
this README can be re-run from that list.

Matchups involving `pimc` cost about 250 ms a decision, so they are run over fewer
seeds; everything else is comfortable at 200-500.

### Diagnostics

```powershell
.\.venv\Scripts\python scripts\smoke.py 500 --greedy   # what a batch of games looks like
.\.venv\Scripts\python scripts\calibrate_stakes.py     # payout sensitivity check
.\.venv\Scripts\python scripts\detect_device.py        # what this machine will train on
```

`calibrate_stakes.py` re-runs the game with every payout scaled up and down. Two
uses: confirm the simulated game *length* at 1.0× matches a real round, and see how
sensitive the endings are to the table being slightly off. Currently bankruptcy runs
7% → 13% → 28% across 0.8× → 1.0× → 1.25×, so a small error in the table shifts the
balance noticeably without changing the character of the game.

M2 is the gate. The table is built; what remains is playing a real round beside it.
No training compute gets spent until that comparison is done, because a rules error
found at M6 costs a retrain and one found now costs a YAML edit.

### The one thing still assumed

Everything else has been confirmed against real play. What remains is **which 100
of the possible cards are in the deck** — and it cannot be resolved by looking.

The game shows a "remaining cards" list, but it counts every card the player has
not *seen*, drawn from the full theoretical pool of 9 copies per holomem — 153
cards for a 17-holomem lineup. The deck only ever holds 100. So about a third of
that list is cards that do not exist in the game at all, and the number it shows is
not the number of cards left.

**This is where the bot's structural edge comes from.** A human reading that list is
being confidently misled and has no practical way not to be. Inferring the real
composition from cards actually observed is not a refinement here; it is the only
way to know. That is what the composition posterior in `envs/belief.py` is for, and
`pokajan/vision/` carries a note never to scrape the counter as truth.

As of M3 that claim is measured rather than asserted. Against the true deck, the
posterior's estimate of what is left is out by **0.11 cards per slot** where the
counter's reasoning is out by **1.04** — 9.3× better, and the gap is structural
rather than lucky, which is why `tests/scenarios/test_belief_scenarios.py` pins it
as a ratio and fails if the posterior ever stops beating it.

The inference is two levels, and the first is the one humans cannot do at all:

1. **Composition** — how many copies of each card this game was built with.
   Updated per slot from a hypergeometric likelihood, then coupled back together
   so the totals come to exactly 100. That coupling is where "I have seen four of
   these, so something else must be thinner than I thought" lives, and the in-game
   counter cannot express it even in principle.
2. **Location** — given a composition, where the unseen cards sit. Weighted
   particles rather than per-slot averages, because the sharpest evidence is a
   *joint* constraint: the engine only offers a claim to seats that could legally
   make one, so a card left on the table has been declined by everyone who could
   have used it. That rules out whole shapes of hand at once, and a per-slot
   average would blur it away to nothing.

Particles are also exactly what PIMC needs at M4, so the two consumers share one
implementation instead of drifting apart.

### What the heuristic does with it

Everything is priced in coins, never in cards-from-completion. That sounds like a
detail and is not: a hand one card from a monochrome triple (840) and a hand one
card from a mixed one (120) are *identical* by any shanten count, and a hand two
cards from 1800 beats a hand one card from 120. Ranking by distance ranks this game
wrongly. So a hand is worth `payout x P(completing it)`, maximised over every
target the roster admits, with the probability coming from the belief.

One asymmetry is modelled exactly, because getting it wrong inflates every long
shot: **a claim can only ever finish a hand.** Claiming is legal only when the
claimed card completes the hand, so no number of opponent discards moves you from
two short to one short. Only the last card gets the extra chances.

Defence is the other half, and it matters more here than in most card games because
a claim bills **the discarder alone** — there is no pot to share the damage. Each
candidate discard is scored against the particles for what it would cost if
claimed, taking a maximum across opponents rather than a sum, since only one claim
can win. Both halves come out in coins, so the discard rule subtracts one from the
other with no weighting factor to tune. Turning that term off costs 173 coins/game.

Since the underlying rule is unobservable, M5 should train against a *mixture* of
composition rules rather than committing to one — an agent calibrated to the wrong
rule would be wrong in exactly the same confident way the in-game counter is.

## Quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m pytest
```

The overlay is a second window, pinned over whatever is on screen:

```powershell
.\.venv\Scripts\python -m pokajan.server.app          # leave this running
.\.venv\Scripts\python -m pokajan.server.overlay --place   # drag to move, corner to resize
.\.venv\Scripts\python -m pokajan.server.overlay      # pinned, inert
.\.venv\Scripts\python -m pokajan.server.overlay --check   # report what actually applied
```

Frameless, transparent, always on top, and — the part that matters — it takes no
clicks and never takes focus. The real game is played online against real people, so
an advisory panel has to be physically incapable of interfering with input. That is
enforced twice over: `WS_EX_TRANSPARENT | WS_EX_NOACTIVATE` on the window, and a
socket the overlay can only read from.

Being inert is also why placement needs its own mode — a window that takes no clicks
cannot be dragged — and why `--check` exists. Every mechanism here fails *silently*:
window styles that did not apply, a drag region pywebview never bound to, a resize
API that never reached the page. So it reports all of them rather than assuming:

```
pinned: clicks pass through, and it will not take focus.
styles  0x080d0028 [layered, click-through, no-activate]
size    520x190 css at 2x = 1040x380 physical
handles 1 drag region(s), pinned so no grip
```

Training extras are separate, because the two machines this runs on need different
PyTorch builds:

```powershell
.\.venv\Scripts\pip install -r requirements-train.txt
.\.venv\Scripts\python scripts\detect_device.py     # reports what it will train on
```

## The one rule about rules

**`rules/pokajan_v1.yaml` is the only place gameplay numbers live.** Nothing under
`pokajan/core/` may define one.

This is not tidiness — most of the payout table is still a guess. Designing so that
a corrected payout is a one-line YAML edit, rather than a hunt through the code,
is what lets the rest of the project start before the rules are fully known.
Entries are marked `CONFIRMED` or `TODO(M2)` so it is always clear which is which.

Every config has a `rules_hash`, stamped into checkpoints and eval reports. A model
trained under one payout table is meaningless under another, so loading across a
mismatch requires an explicit override.

## Rules, as currently understood

Four players, 1000 coins each, one round. Draw one card, discard one, hand fixed at
seven. A **power hand** is three copies of one character, or one of every member of
a group. Calling it ("Pokajan") collects coins:

- claimed off another player's discard → **that player alone pays**
- otherwise → **the other three split it evenly**

After scoring, the cards leave your hand, you refill to seven, and if the refill
completes another hand you may call again — indefinitely.

### The payout table (confirmed)

|  hand   | multi-colour | single-colour | premium |
|---------|-------------:|--------------:|--------:|
| triple  |          120 |           840 |   7.00× |
| 3-group |          180 |           480 |   2.67× |
| 4-group |          300 |           840 |   2.80× |
| 5-group |          480 |          1800 |   3.75× |

Plus **+90 per copy** of the randomly chosen bonus holomem that the hand scores —
additive, so a triple of the bonus character is +270.

Three things to notice, all of which shaped the code:

- **The monochrome premium is not a constant multiplier**, so payouts are a flat
  table rather than base × modifier. Nothing may infer one row from another.
- **A monochrome triple (840) outranks a monochrome 3-group (480)** and ties a
  monochrome 4-group. Since strength *is* payout, nothing may assume group hands
  beat triples.
- **A monochrome 5-group pays 1800, more than the entire starting stack.** Off a
  discard, that bankrupts the discarder outright and ends the game on the spot.

A discard can be claimed by anyone, out of turn, but only in the instant after it is
played. An out-of-turn claim does not move the turn and costs you no discard. A hand
that completes without a discard can only be called on your own turn, so you can be
stuck sitting on a made hand.

When several players claim the same card, **the larger payout wins**; ties go to the
earliest in turn order. Hand strength is not a separate concept — it *is* the payout.

The game ends when the deck runs out or any player hits zero coins. Highest coin
count wins.

### Three things that make this more interesting than it looks

**Coins are not conserved.** If a payout would take the payer below zero, the caller
still receives the full amount but the payer only falls to zero. The difference is
minted. The test suite asserts `sum(coins) == 4000 + minted` rather than a constant,
precisely so this cannot silently regress.

**Short-stack sniping.** Landing a big hand on a nearly-broke player pays you in
full, costs them only what they have left, and ends the game immediately.

**Both endings are player-controllable.** Chained calls refill from the shared deck,
so a long chain can run it dry on demand. A leading player therefore has two
distinct ways to end the game while ahead — which is exactly the kind of decision
the max-EV and safe agents should disagree about.

## Layout

```
rules/pokajan_v1.yaml     every rule number, and nothing else
pokajan/core/             cards, rules, hand evaluation, action space
pokajan/envs/             environment, observations, belief, determinize (M1/M3/M4)
pokajan/agents/           heuristic, fast, PIMC, neural             (M3-M6)
pokajan/train/            evaluation now; behaviour cloning, PPO    (M5+)
pokajan/server/           protocol + web app                        (M2)
pokajan/vision/           screen reading                            (M8)
web/                      browser UI and overlay                    (M2/M7)
data/captures/            real-game payout observations and card art
tests/                    invariants (any config) + scenarios (frozen fixture)
```

Three files are load-bearing beyond their size:

- `pokajan/core/cards.py` fixes the count-vector layout. Everything downstream
  assumes it, and changing it invalidates every trained checkpoint.
- `pokajan/envs/obs.py` fixes the observation layout, for the same reason. It
  carries an `ObsSpec.signature` that gets stamped alongside the rules hash, so a
  checkpoint cannot be loaded against a layout it was not trained under.
- `pokajan/server/protocol.py` defines the only language spoken between "a game"
  and "something that plays it". At M8 the screen reader becomes just another
  producer of `PublicState`, and nothing else has to change.

### Things tried that did not work

Kept because a measured negative is worth more than an untested idea, and each is
one flag away from being re-run against a better model.

**Perfect-Information Monte Carlo (all of M4).** Sample worlds from the belief,
play every candidate action to the end of the game in each, take the action that
averages most coins. It came out at **−144 coins/game** (95% CI [−197, −92])
against the heuristic it wraps, at 250 ms a decision.

The interesting part is why, because it is a fact about Pokajan rather than about
the implementation. Instrumenting real decisions showed the search overruling the
heuristic on 62% of discards and agreeing on every call and chain, so the damage
was entirely in discard selection. Measuring the signal directly:

| | |
|---|---|
| mean difference between the top two discards | **29 coins** |
| standard deviation of that difference, per world | **279 coins** |
| standard error at 16 determinizations | **70 coins** — 2.4× the effect |
| worlds where both discards ended identically | 67% |
| determinizations needed to resolve the effect | **~481** (≈8 s/decision) |

Changing a discard can flip whether somebody claims it; a claim changes how many
cards get refilled, which shifts every subsequent draw for everyone. Two thirds of
the time nothing diverges and the paired comparison is clean, but the other third
explodes, and that tail is thirty times larger than the effect being measured. The
search was picking among the heuristic's top three essentially at random, and
replacing an informative ranking with noise costs exactly what you would expect.

What was ruled out along the way, so the conclusion stands on evidence:

- *A broken determinizer* — no. Handed the true hidden state, it reconstructs the
  real engine exactly across 1200 decisions and every state field; that is now an
  invariant test.
- *A weak rollout policy* — no. Given perfect information it beats the heuristic by
  **+519 coins/game**.
- *A prior handicap* — mostly no. PIMC's internal heuristic runs on 16 belief
  particles rather than 48, worth −38 (95% CI [−89, +13], not significant).
- *Rollout chaos* — partly. Truncating rollouts to 12 rounds recovers about a third
  of the gap (−144 → −85) and does not rescue it.
- *Searching the wrong decisions* — searching only calls, claims and chains and
  leaving discards to the heuristic lands at −68 (CI [−141, +5], not significant),
  i.e. level with its own prior. Search adds nothing there either.

More determinizations is the obvious remedy and the measurement prices it: ~481 per
decision, thirty times the current budget, for one decision type. That is not a
tuning problem. It is why the roadmap goes to a learned policy at M5 rather than to
deeper search.

**Replacing the completion probability with a Monte-Carlo one.** The heuristic
prices a hand with a crude closed form — requirements treated as independent, a
hand-tuned claim-efficiency constant, `(1−p)^opportunities`. The obvious upgrade is
to sample futures from the belief instead and take `E[best payout reachable]`,
which also prices two live chances correctly without the double-counting that sank
the idea above. It is cheap (42 ms a decision) and it samples a quantity that does
*not* diverge chaotically, so it avoids what killed PIMC.

It came out at **−19 coins/game** (95% CI [−69, +32]) — level with the closed form,
not better. Worth keeping for the one thing it did teach: the first version scored
−132, and the whole difference was a horizon cap. Handed every card it will draw
for the rest of the game, a sampled hand can assemble almost any target, because
nothing in the sample makes it discard down to seven. Modelling the future without
modelling the hand limit measures what is in the deck rather than what the hand can
become. The sweep is recorded on `DEFAULT_HORIZON`.

**Valuing a hand by combining its best few targets** instead of taking the best
one. A maximum cannot see that two live chances beat one, which looked like a clear
gap. Measured at **−38 coins/game** (95% CI [−89, +13]) over 480 games — targets
overlap too heavily for independence to hold. See `TARGETS_COMBINED`.

### Does more compute help?

Asked directly, with a budget of five seconds per decision — comfortable for a live
hint at M7 or for labelling training data at M5, and unusable inside a PPO loop.
Three levers, all measured against the same heuristic:

| lever | cost/decision | result |
|---|---:|---|
| **belief precision**, 48 → 1024 particles | 24 → 248 ms | **+50** [+16, +84], over 1280 games |
| sampled valuation, 32 futures | 24 → 42 ms | −19 [−69, +32], not significant |
| determinized search, 16 worlds | 24 → 250 ms | −144 [−197, −92] |

Only the first one buys anything, and the reason is visible *before* running any
matchup — which makes it a useful thing to check first next time. Both the belief
and the search produce a noisy estimate of a quantity that differs between
candidate discards. What matters is the ratio:

| estimator | its noise ÷ the spread it must resolve |
|---|---|
| discard danger, 48 particles | **0.38** |
| PIMC rollout value, 16 determinizations | **2.4** |

Both converge with more samples. The difference is the rate at which they arrive
somewhere useful: the danger estimate is already inside the signal and reaches
0.10 for 20× the compute, while the rollout estimate starts at six times worse and
needs about 30× the budget merely to draw level — eight seconds a decision, for one
decision type. Cheap to measure, and it would have priced the whole milestone in an
afternoon.

The broader read, which shapes M5: this game's decisions are dominated by
quantities the heuristic already computes directly — what a hand is worth and what
a discard risks. Depth adds variance rather than insight, and the leverage is in
knowing the cards better, not in looking further ahead.

### What the failed milestone left behind

Chasing the search produced four things that outlive it, three of which M5 needs:

- **A search fast path on the engine.** `pending_seats()` / `apply()` answer the
  same questions as `pending_decisions()` / `submit()` without building a
  `PublicState` per seat — measured **79× cheaper**, and both public methods are
  now implemented in terms of them so there is no second copy of the turn logic.
- **`envs/determinize.py`**, which rebuilds a playable game from a `PublicState`
  and a belief particle *and nothing else*. That constraint is the whole reason it
  is written this way: at M8 the opponent is the real game and there is no engine
  to clone, so anything that cannot search from a reconstructed public view is not
  the agent this project is building.
- **`FastAgent`** — the rollout policy's cheap valuation with belief-supplied
  defence. It gives up 104 coins/game against the heuristic and runs 3× faster
  (6 ms vs 19 ms a decision), which is the trade a self-play loop wants when the
  opponent pool is queried millions of times.
- **A 3.5× faster belief posterior**, from separating the bisection that finds the
  deck-total tilt from the construction of the rows it tilts.

## Testing

Two tiers, and the split is the point:

- **`tests/invariants/`** run against *randomly generated* rules — any roster from
  14 to 19 characters, any group split, any payout scale — and assert only what must
  hold however the real rules turn out. These keep their value when the numbers get
  corrected.
- **`tests/scenarios/`** pin exact expected payouts to `tests/fixtures/rules_v1.yaml`,
  which is frozen and never edited. When real rules land, they get a `rules_v2.yaml`
  and new scenarios; the old ones still pass, proving the change did not disturb
  behaviour already verified.

Since M3 the scenarios also pin *properties* that no payout can express: that the
posterior beats the in-game counter by a margin rather than a hair, and that a card
nobody claimed measurably lowers the odds that anybody could have. Those two run
against the live config on purpose — the size of the decoy is a fact about the real
game, not about a fixture — so they assert ratios, which survive a rules edit.

The fixture deliberately makes a two-member group pay *less* than a triple, so that
nothing in the codebase can quietly assume groups outrank triples.

## Two machines

Development happens on a laptop (CPU-only, AMD iGPU) and training on a desktop with
a GPU. Everything is device-agnostic: the device is resolved at runtime, checkpoints
save CPU-portable, and no path is absolute. `scripts/detect_device.py` reports what
a given machine will actually use.

## Scope note on screen reading (M8)

Pokajan is played online against real people, so the screen reader is **read-only**.
It captures the window, reconstructs the public state, and displays a recommendation
in an overlay. It does not send clicks or keystrokes — a human makes every move.
