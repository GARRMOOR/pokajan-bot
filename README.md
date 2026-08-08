# pokajan-bot

A learning agent for **Pokajan**, the four-player card minigame in *Hololive Dreams*.

Two agents are the goal: one that maximises final coins, and a safer one that
maximises the probability of finishing above the 1000-coin starting stack. Getting
there means first building a playable copy of the game to train against, then
reading the real game's screen so the bot can advise during live play.

## Status

**M0 — core model.** Rules config, card model, hand evaluator, action space, and
the JSON protocol are in, with 38 tests passing. No engine yet.

```
M0  core model, protocol, tests            <- done
M0b capture real payouts + card art        <- in progress (see data/captures/)
M1  engine + environment
M2  web GUI, human-playable  <-- rules get validated against the real game here
M3  observation encoder, belief, heuristic agent
M4  PIMC agent
M5  vectorised env, behaviour cloning, PPO self-play
M6  risk-conditioned training
M7  hint mode + overlay
M8  screen reading
```

M2 is the gate. No training compute gets spent until a human has played a full
game here and compared it turn-by-turn with the real thing.

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

All-one-colour hands pay more, as does a hand containing the game's randomly chosen
bonus character. After scoring, the cards leave your hand, you refill to seven, and
if the refill completes another hand you may call again — indefinitely.

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
