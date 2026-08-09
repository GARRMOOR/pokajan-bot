# pokajan-bot — working notes

**Read `README.md` first.** It carries current status, the confirmed rules, the
payout table, and every measured result. This file holds only what is easy to get
wrong and not obvious from reading the code.

## Environment

- **Python 3.12 in `.venv`.** The system `python` is 3.14 and has no torch wheels.
- `.\.venv\Scripts\python -m pytest` — 259 tests, ~1 min.
- `.\.venv\Scripts\python -m pokajan.train.evaluate --agent X --baseline Y --seeds 200 --workers 6`
- `.\.venv\Scripts\python -m pokajan.server.app` — playable web table on :8000.
- `.\.venv\Scripts\python -m pokajan.server.overlay [--place|--check]` — the overlay.
- **Vision is checked by scripts, not tests** (see Test strategy):
  `scripts\check_vision.py` scores card and colour reading on real frames;
  `scripts\check_layout.py` draws every region back onto a frame — the only way to
  verify a fraction, and it now also reports the roster each frame reads;
  `scripts\harvest_digits.py` re-cuts the digit exemplars and
  `scripts\harvest_group_labels.py` the group badges. **Both harvesters must be re-run
  and their reads checked back against their own transcriptions when a case is added** —
  the glyph-count check only catches a case with the wrong *number* of digits, and a case
  transcribed as 1040 when the screen said 2270 quietly fed a `2` into the `1` exemplar
  for weeks.
- **numpy belongs to `pokajan/train/` and `pokajan/vision/` only** — the first via
  `requirements-train.txt`, the second via `requirements-vision.txt`. The belief, the
  observation encoder and every agent run on plain lists on purpose: numpy's per-call
  overhead dominates on arrays of fifty-odd elements, and that is measured, not
  aesthetic. Vision is the opposite case, where the array operation *is* the work.
  torch stays training-only and is not installed.
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
   The **inverse also holds**: derived *text* under `data/captures/` is committed on
   purpose — `digits.yaml` (glyph shapes from the game's typeface),
   `hololive_groups.yaml` (public agency membership), `payouts_observed.yaml` and
   `rounds_observed.yaml`. They carry no personal data and they are what lets the
   reader work on a machine that has no screenshots. Keep them plain text so a bad
   entry shows up in a diff.

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
  A test that cannot fail is worse than a missing one, and both later cases were
  subtler than that one: a test of cache-key collisions that picked two seeds whose
  keys did not actually collide, and a test of idempotence that the cache satisfied
  without the code under test ever running twice. Neither looked wrong. **For any test
  guarding a specific regression, break the code and watch it fail** — cheap, and it
  is the only thing that distinguishes a guard from decoration.

- **The advisor is the only thing that talks to a human**, and its failure mode is a
  plausible sentence rather than a crash. Never let generated text imply more
  certainty than the number behind it: `_confidence` measures only whether the
  ranking survives resampling the belief, so below `TOSS_UP` it must say "coin flip"
  and name no deciding reason. Opening discards genuinely are toss-ups — six draws at
  1024 particles on one position gave three different answers.

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
2. ~~Overlay window~~ — **done.** `pokajan/server/overlay.py`. **Everything about
   pywebview here fails silently, and almost nothing behaves as documented** — five
   separate traps, all written up in that module, all reported by `--check` so a
   regression is visible:
   transparency is a **pure-red colour key, not alpha**;
   `focus=False` **does not survive WebView2 startup**;
   `easy_drag` **never binds in 5.3.2** (`util.py` emits `'true'`, its JS tests
   `'True'`);
   a `js_api` object holding a **public** reference to the Window kills startup,
   because pywebview walks it and calls `evaluate_js` too early;
   and `create_window(width=)` **does not round-trip** with `window.width` while
   `resize()` does — they differ by `devicePixelRatio`, so storing the wrong one grew
   the window by the display scale on every launch.
   **Verify against `--check` before believing any of it changed.**
3. ~~Per-game roster construction~~ — **done.** `core/roster.py`: `rules_for_roster`
   edits three keys and re-parses, so a misread roster can never alter what a hand
   pays. It refuses rather than repairs, because a wrong roster produces advice that
   looks exactly like right advice. **Do the same when the reader lands** — every
   validation there exists because the failure is silent.
4. Static recognition from one screenshot → roster, groups, hand, coins. **Cards and
   layout are done.** `vision/layout.py` trims the letterbox and holds every region as a
   fraction; `vision/geometry.py` finds and splits rows and reads frame colour;
   `vision/templates.py` names the holomem. `scripts/check_vision.py` scores 11/12
   holomem and 12/12 colours on a real frame, **0 wrong**, ~10 ms/card.
   **Verify regions with `scripts/check_layout.py`, which draws them back onto a
   frame** — it caught four misplaced boxes first time, one of which put the bonus card
   inside the group panel. A fraction cannot be checked by reading it.
   Nothing here may guess: `identify` refuses on a thin *margin* rather than a low
   score, because an unknown holomem still produces a plausible best match, and an
   incomplete catalogue is the normal state.
   **Digits work too** (`vision/digits.py`), 18/18 on transcribed coin totals and deck
   counters, exemplars in the committed `data/captures/digits.yaml`. Every failure there
   was segmentation, never a mismatched glyph — see that module, and note that the boxes
   in `layout.py` and the thresholds in `digits.py` are two halves of one decision.
   **The digit exemplars are missing a 5**, so any number containing one is refused —
   safe, and not blocking anything yet. It needs a capture with a 5 in a coin total or
   the deck counter. The Gen 5 card badge and the group panel's row labels were both
   tried and neither works: the badge's white body fragments around its pink stroke.
   **Do not read the roster from the panel's portraits.** Card art matched against them
   scores 1/17, all far under the refusal threshold — a crop step is not enough. Read
   the four group *labels* instead and look their members up: the game uses real
   hololive branches, every observed group size is canonical, and a four-way
   classification plus a cell count beats recognising twenty portraits. See
   `pokajan/vision/__init__.py`.
   **Roster-from-panel is done** without recognising a portrait: `vision/roster_panel.py`
   counts each row's real cells (placeholders are flat grey — no recognition), and the
   four group labels are looked up in the committed `data/captures/hololive_groups.yaml`.
   The count is the *cross-check* on the label and it refuses on disagreement; note it
   only catches a label misread as a **differently sized** group.
   A count that is not 3, 4 or 5 means the panel is occluded — a payout frame measured
   [4,3,2,2] between two clean [4,4,4,5]s. Free detector, no extra machinery.
   Card-art filenames drift, so punctuation is normalised away and anything still
   unrecognised is **reported, never fuzzy-matched** — one file was once `usada_pekore`,
   a single letter from a real holomem. Reporting got all of them fixed at source in one
   message, which is the workflow: tell the user the filename.
   **The table excludes graduated members** — Gen1 and Gen2 are fours, not fives — which
   was inferred from the art the user holds against the list of what they still lack, and
   is consistent with Gen3 (no Rushia) and Gen4 (no Coco) at four. Coverage now matches
   their missing list exactly, 16 for 16, which is the check that the whole table is right.
   Still to do: classifying the four labels themselves, ranks, and the
   newest-card-per-seat read in step 5.
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

## Working with the user

They are the only source of game data, and they respond fast to **specific** requests —
name the exact file, value or screen. Two open asks, neither blocking:

- **frames showing the seven group badges never yet captured** — `0` (Gen0), `3` (Gen3),
  `2ID` (ID Gen2), `X` (holoX), `Pr` (Promise), `Ad` (Advent), `Re` (ReGLOSS). A round
  dealing one of those cannot have its roster read at all. `scripts\harvest_group_labels.py`
  prints the current list every run.
- **more card art**, currently 46 of 62 holomem. Coverage decides how often the reader
  names a card rather than refusing.

Closed: **the digit 5**, which appears in no number on the table — every payout is a
multiple of ten, so no coin total ends in one. It was cut from the Gen5 group *badge*
instead, the badges being drawn in the same typeface (measured: coin-harvested exemplars
match the Gen1–Gen4 badges at 0.86–0.97). Still worth confirming from a real coin total if
one ever shows a 5, since it is the one exemplar reasoned across contexts.

They fix misnamed art at source when told, and have offered to let us rename files
directly — either is fine, but **say which**. Do not fuzzy-match a filename to avoid
asking; that is how `usada_pekore` would have become the wrong holomem.

They run every `git commit` themselves. Hand over the message.

## Do this before M5 spends any compute

**The observation vector's length depends on the roster** — 845 dims at 14 holomem,
1130 at 19 — because every block in `envs/obs.py` is sized from `n_slots`. The engine,
belief and heuristic do not care, which is why the overlay works without training
anything. A network cares completely: one trained on a 17-holomem game cannot be fed a
14-holomem observation, and the real game redraws its roster every round.

Pad to the maximum roster with a validity mask. ~10% wasted width, and checkpoints
become portable across rosters instead of worthless. **It is free right now and never
again** — changing the layout invalidates every trained checkpoint, and there are
currently none.

## Test strategy

- **Vision tests are synthetic, always.** The card art and the captures both live under
  `data/` and are gitignored, so no test may depend on them — a fresh clone must pass.
  Generate images in the test. Accuracy against the *real* thing is measured by the
  scripts above, run on the machine that holds the data, and the numbers get written into
  the README. This split is not a compromise: the synthetic tests caught a real bug the
  captures were hiding, where a brightness range computed over the wrong subset made a
  region read as blank whenever antialiasing did not save it.
- **`tests/invariants/`** run against *randomly generated* rosters (14–19 characters,
  any group split, any payout scale) and assert only what must hold under any config.
  These keep their value when numbers get corrected.
- **`tests/scenarios/`** pin exact numbers to the frozen `tests/fixtures/rules_v1.yaml`.
  Payout **values** there are frozen forever; play **mechanics** track the real game.
  A corrected payout gets a `rules_v2.yaml` and new scenarios, never an edit.
