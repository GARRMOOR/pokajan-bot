# pokajan-bot — working notes

**Read `README.md` first.** It carries current status, the confirmed rules, the
payout table, and every measured result. This file holds only what is easy to get
wrong and not obvious from reading the code.

## Environment

- **Python 3.12 in `.venv`.** The system `python` is 3.14 and has no torch wheels.
- `.\.venv\Scripts\python -m pytest` — 142 tests, ~2 min.
- `.\.venv\Scripts\python -m pokajan.train.evaluate --agent X --baseline Y --seeds 200 --workers 6`
- `.\.venv\Scripts\python -m pokajan.server.app` — playable web table on :8000.
- **numpy and torch are training-only** (`requirements-train.txt`) and *not
  installed*. Nothing outside `pokajan/train/` may import them — the belief, the
  observation encoder and every agent run on plain lists on purpose.
- Two machines: this laptop is CPU-only (AMD 780M). The desktop has an **RX 7800 XT
  — AMD, so no CUDA.** On Windows that means DirectML or ZLUDA, not the CUDA path
  the plan's "hours instead of days" estimate assumed. Worth pricing before M5
  commits to a training budget; `scripts/detect_device.py` reports what a machine
  will actually use.

## Hard rules

1. **Never run `git commit`.** Stage the work, then hand the user the message to run
   themselves.
2. **One branch and one commit per milestone** (`M1`, `M2`, `M3`…), each cut from the
   *previous milestone branch*, never from `main` — `main` is still at M0. Check what
   the current branch actually contains before starting.
3. **`rules/pokajan_v1.yaml` is the only place gameplay numbers live.** Nothing under
   `pokajan/core/` may define one. Agent tuning constants are *policy*, not rules,
   and belong in the agent module with a comment saying so.
4. **Never commit anything image-shaped under `data/`.** Those screenshots are of
   live online games and carry real players' usernames. `.gitignore` enforces it.
   Transcribe observations into `data/captures/*.yaml` instead — that is the durable
   artifact and it holds no personal data.

## Architecture that is expensive to reverse

- **`pokajan/server/protocol.py` is the seam.** `PublicState` contains only what a
  seat may legitimately see. At M8 the screen reader becomes just another producer
  of it and nothing else changes.
- **Search must work from `PublicState` + a belief particle alone**, never by cloning
  the live engine. Otherwise it cannot run against the real game, which is the whole
  point. See `envs/determinize.py`.
- `core/cards.py` fixes the count-vector layout; `envs/obs.py` fixes the observation
  layout and stamps `ObsSpec.signature`. Changing either invalidates every trained
  checkpoint.
- The engine has **two surfaces**: `pending_decisions()`/`submit()` for clients, and
  `pending_seats()`/`apply()` for search — 79× cheaper because it builds no
  `PublicState`. Both delegate to the same turn logic.

## Traps

- **Hand strength *is* the coin payout.** Nothing may assume group hands beat
  triples: a monochrome triple (840) beats a monochrome 3-group (480) and *ties* a
  monochrome 4-group.
- **Coins are not conserved.** The asymmetric bankruptcy floor mints them. Assert
  `sum(coins) == players * initial + minted`, never a constant. Confirmed in real
  play: an observed game ended 720 + 1040 + 2270 + 0 = 4030.
- **A claim must actually SPEND the claimed card** (`evaluate.can_call_using`). The
  naive "would my hand score with this added" is trivially true for an already-made
  hand and hands the agent a large illegitimate edge.
- **A call refills to your pre-discard size**, not flat to the hand limit. Refilling
  flat costs a card permanently, every time.
- **Claim windows must ask every eligible seat against one frozen snapshot**, with
  nothing mutating until all have answered.
- **Vacuous assertions have slipped into this suite before** (`assert x == y or True`).
  A test that cannot fail is worse than a missing one.

## Do not re-try these — they are measured negatives

Each survives as a registered agent in `train/evaluate.py` so the finding can be
rechecked rather than taken on trust.

| tried | result |
|---|---|
| PIMC / determinized search | **−144 coins/game**. Rollout noise is 2.4× the signal; ~481 determinizations (≈8 s/decision) would be needed. |
| Monte-Carlo hand valuation | **−19** (level with the closed form). And −132 without the horizon cap. |
| Combining a hand's best four targets instead of the max | **−38** |
| **More belief particles** | **+50** — the one thing that worked (`heuristic-p1024`) |

The general lesson, worth checking before spending compute anywhere: compare an
estimator's sampling noise against the spread it must resolve. Below 1 and more
samples help cheaply; well above 1 they cannot rescue it.

## The overlay does not depend on M5 or M6

The plan runs M5 (training) → M6 (risk) → M7 (overlay) → M8 (vision), but that
ordering is about building the *best* agent, not about getting a usable overlay. The
overlay's real dependency is only M7 + M8: the heuristic already plays well and
already decides from a `PublicState`, and because `protocol.py` is the seam, swapping
in a learned agent later changes nothing else. **Do not assume training has to come
first** — it is roughly two milestones of delay for no benefit to the overlay.

Order that actually works, with the risk concentrated late:

1. ~~Hint panel in the existing web UI~~ — **done.** `agents/advisor.py` is the only
   producer of `Recommendation`; `Session.hint()` serves it on request; `web/app.js`
   renders it. Measured at 16 ms (48 samples) to 210 ms (1024) against a ~10 s turn.
2. Overlay window (frameless, transparent, click-through — pywebview gives the first
   three, click-through needs a Win32 `WS_EX_TRANSPARENT` call via ctypes).
3. Per-game roster construction: `rules/pokajan_v1.yaml` pins one roster, but the real
   game redraws 14–19 characters and 4 groups every round. Everything downstream is
   roster-agnostic, so this is "build a `Rules` from the observed roster", not a
   refactor — but nothing does it yet and M8 cannot start without it. Cheaper than it
   looks: the roster panel stays on the table all round, so there is no animation to
   catch and no state to reconstruct.
4. Static recognition from one screenshot → roster, groups, hand, coins.
5. Event tracking across a live round, with a "lost track — no advice" guard. **This
   is where the real risk is**: `table` and `scored` must be accumulated by watching
   continuously, so one missed claim silently corrupts the belief, which looks like
   bad advice rather than a bug. Captures confirm there is no shortcut — older
   discards stack until neither the cards nor their count can be read, so no single
   frame lets a reader catch up. See `pokajan/vision/__init__.py`.

The browser panel already lives under that same constraint on purpose:
`Session._snapshot` feeds `Advisor.observe` on **every** state, not only the ones a
hint is asked about, because an unclaimed discard is visible only as a difference
between consecutive views. Moving that into `hint()` for speed would leave the
belief inferring from gaps and nothing would look broken — there is a test.

## Test strategy

- **`tests/invariants/`** run against *randomly generated* rosters (14–19 characters,
  any group split, any payout scale) and assert only what must hold under any config.
  These keep their value when numbers get corrected.
- **`tests/scenarios/`** pin exact numbers to the frozen `tests/fixtures/rules_v1.yaml`.
  Payout **values** there are frozen forever; play **mechanics** track the real game.
  A corrected payout gets a `rules_v2.yaml` and new scenarios, never an edit.
