# pokajan-bot

A learning agent for **Pokajan**, the four-player card minigame in *Hololive Dreams*.

Two agents are the goal: one that maximises final coins, and a safer one that
maximises the probability of finishing above the 1000-coin starting stack. Getting
there means first building a playable copy of the game to train against, then
reading the real game's screen so the bot can advise during live play.

## Status

**M2 — playable in a browser.** Full game logic, an AEC environment, the confirmed
payout table, and a web table you can sit down and play, with 80 tests passing.
Games run roughly 43 turns and 11 Pokajans under greedy self-play, ending 86% on
deck exhaustion and 14% on bankruptcy.

```powershell
.\.venv\Scripts\python -m pokajan.server.app     # then open http://127.0.0.1:8000
```

You take a seat, simple bots take the other three. The point is not the game — it
is the three panels around it: every payout is shown with the arithmetic that
produced it, the transcript reads like something you can hold next to a real round
and check line by line, and the *Still guessing* tab lists the rules we are
assuming rather than knowing. Play a real round alongside it and those go away.

```
M0  core model, protocol, tests            <- done
M1  engine + environment                   <- done
M2  web GUI, human-playable                <- done
M0b capture real payouts + card art        <- payouts confirmed; card art still open
M3  observation encoder, belief, heuristic agent
M4  PIMC agent
M5  vectorised env, behaviour cloning, PPO self-play
M6  risk-conditioned training
M7  hint mode + overlay
M8  screen reading
```

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

Since the underlying rule is unobservable, M5 should train against a *mixture* of
composition rules rather than committing to one — an agent calibrated to the wrong
rule would be wrong in exactly the same confident way the in-game counter is.

## Quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m pytest
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
pokajan/envs/             environment, observations, belief         (M1/M3)
pokajan/agents/           heuristic, PIMC, neural                   (M3-M6)
pokajan/train/            behaviour cloning, PPO, evaluation        (M5+)
pokajan/server/           protocol + web app                        (M2)
pokajan/vision/           screen reading                            (M8)
web/                      browser UI and overlay                    (M2/M7)
data/captures/            real-game payout observations and card art
tests/                    invariants (any config) + scenarios (frozen fixture)
```

Two files are load-bearing beyond their size:

- `pokajan/core/cards.py` fixes the count-vector layout. Everything downstream
  assumes it, and changing it invalidates every trained checkpoint.
- `pokajan/server/protocol.py` defines the only language spoken between "a game"
  and "something that plays it". At M8 the screen reader becomes just another
  producer of `PublicState`, and nothing else has to change.

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
