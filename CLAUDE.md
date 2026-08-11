# pokajan-bot — working notes

**Read `README.md` first.** It carries current status, the confirmed rules, the
payout table, and every measured result. This file holds only what is easy to get
wrong and not obvious from reading the code.

## Environment

- **Python 3.12 in `.venv`.** The system `python` is 3.14 and has no torch wheels.
- `.\.venv\Scripts\python -m pytest` — 333 tests, ~30 s.
- `.\.venv\Scripts\python scripts\capture.py` — watch the game and log what it reads.
  `--frame <name>` reads a saved capture instead, which is how the pipeline gets exercised
  without the game running; `--status` reports disk use; `--purge` deletes every kept crop.
- `.\.venv\Scripts\python -m pokajan.train.evaluate --agent X --baseline Y --seeds 200 --workers 6`
- `.\.venv\Scripts\python -m pokajan.server.app` — playable web table on :8000.
- `.\.venv\Scripts\python -m pokajan.server.overlay [--place|--check]` — the overlay.
- **Vision is checked by scripts, not tests** (see Test strategy):
  `scripts\check_vision.py` scores card and colour reading on real frames;
  `scripts\check_layout.py` draws every region back onto a frame — the only way to
  verify a fraction, and it now also reports the roster each frame reads;
  `scripts\harvest_digits.py` re-cuts the digit exemplars and
  `scripts\capture.py` watches the live game and writes `data/games/*.jsonl` — no screenshot
  ever reaches disk; `--tune` reports why a live read fails without saving anything, `--status`
  what is on disk, `--purge` clears the crop store, `--frame <path>` replays a saved capture;
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
   **The branch is currently `M4` and carries commits M7a–M7j, M8a and M8c**, so the
   rule and the repo disagree. Flagged to the user more than once and left alone
   deliberately: renaming or re-cutting is their call, not something to tidy up
   unasked. Do not be surprised by it, and do not "fix" it.
3. **`rules/pokajan_v1.yaml` is the only place gameplay numbers live.** Nothing under
   `pokajan/core/` may define one. Agent tuning constants are *policy*, not rules,
   and belong in the agent module with a comment saying so.
4. **Never commit anything image-shaped under `data/`.** Those screenshots are of
   live online games and carry real players' usernames. `.gitignore` enforces it.
   Transcribe observations into `data/captures/*.yaml` instead — that is the durable
   artifact and it holds no personal data.
   The **inverse also holds**: derived *text* under `data/captures/` is committed on
   purpose — `digits.yaml` (glyph shapes from the game's typeface),
   `group_labels.yaml` (badge glyphs), `hololive_groups.yaml` (public agency
   membership), `payouts_observed.yaml` and `rounds_observed.yaml`. They carry no
   personal data and they are what lets the reader work on a machine that has no
   screenshots. Keep them plain text so a bad entry shows up in a diff.
5. **`scripts/capture.py` writes no screenshot, ever.** A frame is grabbed into memory,
   read, and dropped — there is no moment at which a picture of a live match is a file.
   Stronger than deleting one afterwards, and the player asked for it: this machine is
   short of disk. The history is `data/games/*.jsonl` instead, which is why every
   refusal is written down — anything the reader missed is gone with the frame.
   The one exception is `data/pending/`: small **crops** of regions whose readers are
   not built yet, under a byte cap, oldest evicted first, `--purge` to clear.
   `capture.CROPPABLE` is an **allowlist** and must stay one — a blocklist fails open,
   and the region that would hurt is `COINS`, whose box deliberately reaches up over
   the player's name so `digits.split_digits` can find the number band beneath it.
   Delete the store and the allowlist when those readers land.

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

- **A threshold tuned on Steam screenshots does not necessarily transfer to a live grab of
  the same pixels.** Every capture in `data/tables/` is a Steam JPEG; the watcher grabs the
  compositor. The coin icon sits inside `COINS["bottom"]` and always did — on the JPEGs it is
  saturated enough that the pale-ink filter shreds it into 8–16px fragments the height filter
  drops, so the box *looked* like it excluded the icon. Live, the same icon renders less
  saturated, survives as two 54–56px "glyphs", and `coins_bottom` refused on **every frame of
  a real session**. Exclude by **geometry**, which is durable; relying on a colour filter to
  remove something is a bet on the capture path. Anything calibrated only against
  `data/tables/` should be re-checked live before it is trusted — `--tune` reports the
  segmentation as text, which matters because those boxes contain a username and cannot be
  cropped for inspection.

- **A card read must name an id the engine knows.** `templates.character_from_filename`
  produces the catalogue's keys and they have to be canonical ids, punctuation and all folded
  away — otherwise `identify` returns a name that is not a character in the loaded `Rules`
  and the card cannot be turned into a slot at all. That was broken for exactly one holomem
  and invisible for weeks, because the file that would have exposed it did not exist:
  `ninomae_ina'nis` kept its apostrophe. It surfaced the moment the art arrived that
  completed the catalogue, as 352 of 1365 rosters reporting a holomem with no art while
  coverage read 62 of 62. **Naming something the engine has never heard of is worse than
  refusing**, because a refusal is handled. Two normalisers now do this in different modules
  and a test pins them together.

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

On the vision side, same rule, three more negatives:

| tried | result |
|---|---|
| Matching a group badge as one picture | "2ID" vs "3ID" **0.14 margin**, under the 0.15 threshold — and both groups have three members, so the member count cannot break the tie. Split the primary from the subscript: 0.26. |
| Raising the badge canvas 28px → 96px to fix that | **No effect** (0.136 → 0.121). Resolution does not change a ratio of shared to distinguishing ink. |
| Rectifying a card's aspect before matching | **No effect at all** — `templates.query_variants` already resizes every query to fixed dimensions, so the input aspect is discarded before matching. Aspect matters for *segmentation*, never for the match. |
| Expecting `FRAME_REFERENCES` to need live recalibration | **15/15 live cards correct**, distances 10–33 against a 120 limit. The Steam-JPEG-vs-live-grab problem that broke the coin boxes does *not* extend to colour. |
| Letterboxing a glyph to preserve its aspect | **Worse at every size** 12x18 to 36x36 — mean pairwise 0.31 vs 0.20, tightest real margin 0.15 vs 0.27. Identical padding *correlates*, dragging every pair toward 1 together. Stretch instead. |
| Matching panel portraits against card art | **1/17.** They are a different rendering, not a crop; hence reading the badges at all. |
| Reading a badge with `digits.DigitReader` | "Ga" → a **confident 0** at 0.74, runner-up 0.25 behind. A digit alphabet has nothing for a G to lose to. |
| Gating captures on the table holding still | **2 of 77** consecutive frame pairs looked settled over 45s of live play; worst movement in *every* region exceeded 190/255. Six records in a round where the deck fell 53→20. Something is always moving. Use the deck counter's **value** instead — 1.7 ms, and it is the game's own turn clock. `--tune` is what measured this. |
| Quantising a change signature before comparing | No tolerance at a bucket boundary, so +1 to every pixel reads as a change. Threshold the difference instead. |
| Settling the coin ledger on the first non-zero change | The four seats are not read on the same frame, so a payout arrives piecemeal and half of one looks like a small one. **8 unexplained events, neither round balancing.** Hold the candidate open until the change forms a legal payout: 21 events, both rounds closing at 4000 + minted. |
| Accumulating `table` by appending the newest discard each turn | Needs every turn seen; the deck counter fell one at a time on only **17/81 and 11/59** draws. Four in five turns pass unobserved. Merge by `max` per card into a lower bound instead — no alignment needed, and it cannot be wrong, only short. |
| Aligning overlapping views of a discard row by sequence | Ambiguous exactly where it matters: `watame, watame, gura` overlaps a later `watame, gura, polka` two ways, and the wrong choice puts a card nothing saw into `table`. |
| Calling a box "verified on two frames" | Both captures of the payout meld turned out to be **the same caller**, and the meld moves with the caller. Two samples of one condition look exactly like two samples. Check what *varies* between your examples before counting them. |
| Surveying only while the coin ledger is mid-payout | 17 frames of 114, mostly the permanent furniture, and both bottom-seat calls missed. The window is narrower than the animation. Survey every read frame — 54 ms × 114 is six seconds a round — and the ordinary frames double as the furniture baseline. |
| Reporting only the score when a read is refused on score | A whole round logged "amelia_watson only scored 0.42" 68 times and there was no way to tell a correct read under the floor from a coin-flip — in the module whose docstring says the *margin* decides. Put both numbers in every refusal. |
| Predicting the fourth meld box from the other three by symmetry | Off by **half a card** in y. It then named the holomem correctly three times over and returned blue/blue/None against a true blue/pink/pink — a triple is three of one holomem, so identification cannot detect a drifted slice. Only the colours catch it. |
| Widening one meld box to hold five cards and letting `find_row` count | The extra room reaches the deck pile and the decoy card list, which merge with the meld into a single stretch and take the count with them: `runs (1, 4)` and an 8px "card". |
| Segmenting a meld at all, at any box size | The same merge defeats exactly-sized boxes too once a meld passes three cards, and a payout rains orange coins across the table so colour cannot separate them either. **Every group call was lost this way** — read as the rightmost three of itself. The geometry is fully determined, so slice it and skip detection entirely. |
| Believing the watcher polls at 2–3 Hz | It manages **0.45 Hz**. Measured from consecutive `at` stamps in a live round: 2.2 s between frames at best, 5.3 s at worst, where this module's notes claimed "about 40 ms of grab". A payout meld is on screen for a second or two, so that rate *is* why four- and five-card melds go unseen. `find_play_area` at full resolution was 178 ms of it (now 54 ms, strided), and twelve meld candidates took a read from 153 ms to 360 ms (now short-circuited). |
| Assuming the rest of that was the screen grab | It was not, and the instrumentation is the only reason we know. Live: **grab 71 ms, BGRA→RGB 0 ms** — together a sixth of one poll of a loop that was taking 2.9 s. Guessing which end of a pipeline is slow has now been wrong twice in a row in this module. |
| Keeping crops on every read frame | **The actual answer, and it cost 2 s a frame — seven times the 285 ms read it was delaying.** Two independent halves. `optimize=True` on four discard crops: 171–247 ms each against 25–68 ms plain, for 9% fewer bytes that buy no disk at all because `max_bytes` caps the store either way. And `files()` walking the directory while `total_bytes()` walked it again: 131 ms + 193 ms per crop against a full 1221-file store, ×4 crops = **1.30 s per frame deciding what to delete**. One `os.scandir` carrying sizes: 17 ms. Crops feed a reader that does not exist yet; they are now rationed by `crops_due` — one set per 15 s, and none at all while `ledger.pending`. |
| Lowering `MIN_SCORE` to catch cards the payout animation dims | **It names furniture.** The left and bottom meld boxes overlap permanent fixtures that match `tokoyami_towa` at 0.30 and `kaela_kovalskia` at 0.42 — the *same* scores on frames minutes apart, which is what a fixture looks like and a card never does. Any floor low enough to catch a real 0.37 card is low enough to accept those. Use the margin as a second way past the floor instead (`CONFIDENT_MARGIN`, with `MIN_FLOOR` as a backstop): measured across all 64 meld-slot refusals ever logged, furniture ran **+0.00 to +0.11** and real cards **+0.19 to +0.24**, with nothing in between. It accepts 7 of those 64 and 85 hand/bonus reads — including one card refused 31 times in a single round — and changes the labelled corpus not at all, since every correct read there already scores 0.53–0.78. |
| Assuming every seat's meld grows the same way | **The bottom seat grows rightward, `right` and `top` grow leftward, and `left` is not known** — and a *three-card* meld is the identical three boxes under either reading, which is why this survived four rounds. The bottom anchor was calibrated on a triple, so every bottom triple read perfectly and every bottom four- and five-card call was cut from bare felt. `right` and `top` were then confirmed on real four-card group calls and looked like proof the rule was universal. Two seats agreeing is not a rule, and a length that *cannot* disagree is not evidence. Settled from the recorded survey on a `+480` five-group and a `+300` four-group: right-anchoring needs cards at 0.260–0.365 and centring needs them from 0.312, and **nothing frame-coloured exists below 0.365** on either frame, while content runs past the old anchor to 0.636. |
| Letting a meld *extension* fail silently | The blind spot that hid the above for four rounds. A longer candidate is only tried because the three-card slice already read, so the seat certainly has a meld and the only question is its length — "the fourth card is bare felt" is the answer, not the absence of one. For the three leftward seats the new card is index 0, which the old `index or colour` rule kept quiet, so a bottom five-group logged as an ordinary triple with **no refusal at all**. Always report an extension. |
| Catching a failed server start with `except Exception` | uvicorn answers a port it cannot bind with **`sys.exit(1)`**, and `SystemExit` derives from `BaseException` — so it sails straight through, the thread dies noisily, and the caller blames a timeout for a bind failure. The port being taken is the *likely* case here (a browser session already serving, or a second watcher), so it is worth catching by name and saying so. |
| Requiring the whole hand on a single frame | Seven cards each refusing independently, so it is the bonus card's fragility with seven chances to fail instead of one. In a bad round a slot reads on **12–38%** of frames; one 34-frame round read every slot at some point, never all seven at once, and produced no advice at all. Assemble it across frames instead (`_merge_hand`) and allow `HAND_UNREAD_MAX` still unnamed — an unread card is definitely still *held*, so the advice is sound, just narrower. The cost is real and is why it is capped: the count vector is short, so a seven-card hand gets priced as six. |
| Merging the hand positionally without checking agreement | The hand **re-sorts** when a card is inserted and everything after it shifts along. Measured across consecutive frames with the deck counter unmoved, **3.85% of shared positions disagree** — so blind positional merging invents a card that was never held about once in twenty-six merges. Merge only when every position both frames read agrees, and drop the candidate outright on a disagreement or a moved deck counter. |
| Judging a round by whether it was tracking on its **last** frame | The last frame is usually mid-animation, and the metric hid a real defect for weeks. The old rule stored only *complete* hands and kept them forever, so end-of-round always looked tracked — while the stored hand was **6 frames stale at the median and 61 at worst**, several turns out of date. Fixing that looked like a regression (end-of-round "tracking" 23 → 13) and was the opposite: frame-level advice held at 49% and rounds advising at all went **24 → 27**. `replay.py` now reports advice availability per frame. |
| Letting a digit misread reach the ledger | A misread coin display is otherwise indistinguishable from a real coin change, and **one costs a whole round**: a `bottom` seat read as 1 while the game-over banner came up put a 138-frame round out of balance by exactly one coin, with nothing else in it wrong. Every coin total is a multiple of 10 and this is a *theorem* — every payout divides by three, every three-way share is a multiple of ten, the stack starts at one, and the floor stops a payer at exactly zero. Derived with `_coin_granularity` rather than written as 10, because the payout table is the one thing still expected to be corrected. Across every logged round it throws out **4 readings of 8391** (1, 13, 4, 1 — all on covered or animating displays) and recovered a round. |
| Expecting the turn indicator to be near the seat it names | It is not on the nameplate or the coin display. It is four bars around the **central oval**, one per seat on the edge *nearest* that seat, with the active one lit yellow. Found by asking for two captures with different seats active and diffing candidate regions rather than by hunting where it seemed it ought to be. Measured on those two captures with the active seat known: the lit bar reads a yellow fraction of 0.33–0.38 and the other three read **exactly 0.00**. Across all 18 captures the lit values run 0.14–0.38 against 0.00–0.01 unlit, and no frame has ever shown two lit. Costs four small crops and a threshold — no matching. |
| Assuming a region you have never seen active is in the wrong place | `bottom` has never been caught lit, since neither capture is the player's own turn. Its box is still *placed* rather than guessed: unlit it reads (45,111,17), matching the other three unlit bars (43,113,12) and not the felt around it (35,123,0), whose blue channel is 0. Whether it lights the same way is the one part a live round still has to confirm — and that is a much smaller claim than "the box is probably right". |
| Treating `rules/pokajan_v1.yaml` as a catalogue of every holomem | It describes **one round** — four groups, seventeen characters — because a slot index only means anything relative to the roster in play, and the roster is redrawn every round. A round read off the screen must build its own `Rules` from the four groups it saw (`vision/state.round_rules`), substituting the roster into the file's own blocks so the payout table, hand limit, deck size and coin floor are still read from the one place gameplay numbers live. Copying any of those into the bridge would fork them, and a corrected payout would then be right in the engine and wrong in the overlay. |
| Topping `table` up to what conservation allows | `envs/belief.py` reads `hand + table + scored` as "cards accounted for" and treats the rest as unseen, so an over-count makes the agent confidently wrong about a card it has already watched leave, while an under-count only makes it less certain. Two seats' staircases are unread and four turns in five pass unobserved, so the bridge is short constantly — it stays short, and reports the gap as `PublicView.short_by` instead of filling it. |
| Settling a round-constant fact on the **first** frame that reads it | The bonus card is the same card all round, so two disagreeing outright reads mean one is a misread — and latching on the first, then calling the second a conflict, lets one bad frame beat every good one. A 141-frame round read the card outright exactly twice, once `himemori_luna` and once `tokoyami_towa`, while **98 frames ranked `himemori_luna` top against 9**. Tally the outright reads and let the round's own rankings break the tie; report a stalemate only when they cannot. A tie-breaker that lands on a holomem *nobody read outright* does not count — that is two channels disagreeing, not one being noisy. |
| A guard clause that reads like a safeguard and is really a short-circuit | `bonus_by_vote` began with `if self.bonus is not None: return None`, which sounds like "an outright read beats a vote" and is fine while the vote is only ever a fallback. The moment the vote is also the tie-breaker *between* outright reads, that line makes it silent in exactly the case it is needed — and worse, unsets an already-settled bonus. Keep such properties pure and put the precedence at the call site. Its test asserted the short-circuit rather than the behaviour, so it passed either way. |
| Believing the cross-check's own diagnosis of who was wrong | "The discard fields show more cards than conservation allows" printed *"the discard fields were misread"*, and it was wrong both times — the error was on the conservation side twice over. Two independent channels disagreeing does not say which one lied. Go and find out. |
| Letting a claimed card stay on the discard floor | A claimed call takes its card **off a pile**, but `DiscardFloor` merges by `max` and never removes anything — rightly, since it cannot tell a claimed card from one buried by a later discard. So the card goes on counting as table while `scored` counts it too: **42 cards across 15 rounds** sit in both the floor and a meld that scored. `table_floor` subtracts one per claimed payout; which card of the meld it was does not matter, because this bounds the count and not the identities. |
| Assuming `in_hands` is always `seats × hand_limit` | Scoring cards leave the hand *before* the deck replaces them, so the caller sits at `limit - k` in between — and when the call is the one that bankrupts somebody, the round ends there and the refill never comes. That made `table` too small by up to five, which then made a correct discard floor look like a misread. Two real rounds were lost that way, both ending on a four-card call that bankrupted a seat. Only the **most recent** call can be outstanding (chaining refills before another call is legal) and the slack is a `max`, not a sum — so a test using two calls of the *same* size cannot tell the two readings apart. |
| Assuming the meld boxes' overlap with `discards_bottom` corrupts the floor | It looks alarming — 61% of the bottom meld's height sits inside `discards_bottom`, and `left`'s meld overlaps it too — but across **75 frames where a bottom or left meld was read, zero of its cards appeared in that frame's discard row**. `find_row` locks onto the actual discard row. A latent hazard worth knowing about, not a bug to fix blind. |
| Triggering the payout sprint from `CoinLedger.pending` | **One signal too late, and it never fired.** Noticing an unexplained coin change means having already sampled it, and the meld and the coin change happen together — so the sprint began, at best, after the thing it was meant to catch. Measured across every logged round: the gap to the next read *while the deck counter was covered* had a median of **2.46 s and a p90 of 5.30 s**, which is `IDLE_HEARTBEAT`, not `PAYOUT_HEARTBEAT`. A five-card left-seat group was lost in exactly that gap — one frame sampled in its whole payout window, showing bare felt. The counter refusing **is** the signal, it costs 1.7 ms, and it arrives on the frame the payout starts: melds are read on **14.4% of covered frames against 3.2% of readable ones**. `TurnGate.urgent` decides it from the key it just computed, told nothing by anyone. **Confirmed across 27 journals split at the change**: the covered-frame gap went median 2.46 → **1.03 s** and p90 5.30 → **1.44 s**, while the readable-frame gap barely moved (2.97 → 2.76 median), so the cost stayed inside the payout window. Frames carrying a meld rose **5.8% → 8.9%**, and the left seat's geometry was settled by a call this caught. |
| Filling in the **fourth** seat's direction from the three you just measured | Done immediately after being burned by exactly this, in the same edit, while writing a docstring that called it measured. `left` was asserted as leftward-growing on nothing but the pattern `right` and `top` set — **and it turned out to grow rightward**, so the guess was wrong as well as unfounded. Fourteen left melds have been read and **every one was three cards** — the length that cannot tell the directions apart — so it is the bottom seat's blind spot untouched. The one frame where a confirmed `+300` four-group coincides with left-band survey content rules *leftward* out by absence (nothing at 0.270–0.323) and leaves rightward hidden inside a merged object. One observation is not a measurement: `read_meld` now cuts **both** directions for any seat in `MELD_UNSETTLED` and lets the payout amount choose, which costs nothing measurable — 241 ms against 246 ms mean read over 18 real frames, because the extra cut only fires once a triple has read there. |
| Reading a survey's per-seat y-band as an upper bound on a meld's extent | The filter that gives clean seat attribution — objects whose y-range matches a meld box — is the filter that *discards* the merged objects a longer meld produces. It reported "no left content right of 0.481, across 39 objects", which looks like proof of leftward growth and is a selection artifact; the same filter finds **zero** objects for the bottom seat, whose direction is settled. Absence arguments only work over the *unfiltered* survey. |
| Testing a per-seat constant against the helper that consumes it | `assert (three.left == five.left) is (seat in MELD_GROWS_RIGHT)` passes for **every** value of `MELD_GROWS_RIGHT` — flipping the seat flips both sides together. It survived the exact regression it was written for. Pin the measured pixels instead: 0.365 and 0.636 for the bottom seat, 0.676/0.630/0.481 for the others. |
| Assuming the roster restriction reaches the reads that need it | It reaches **none** of them. The roster reads on 59% of frames in a live round and on **0% of the frames showing a meld** — every meld fixture on disk too — because the payout panel that reveals a meld is the panel that covers the group list. So the one read where a wrong answer is unrecoverable ranked cards against all 89 holomem. Passing the round's own roster as `roster_hint` leaves every score identical and lifts margins by **+0.00 to +0.17** on the three meld fixtures. Scores identical is the point: `MIN_FLOOR` is untouched and only the margin moves, which is what `CONFIDENT_MARGIN` gates on. |
| A/B-ing a template threshold by re-running `replay.py` | **Cannot work, and it silently reports "no change" rather than failing.** The journal holds derived text and the frames are gone by design, so replay re-runs the *accumulator* over recorded readings — no pixel is ever matched again. Threshold changes are measurable only on the labelled corpus (`check_vision.py`) or on a fresh live round. What recorded data *can* answer is the false-accept question: replay the logged score/margin pairs through the new rule and see what it would newly admit. |
| Fixing the payout heartbeat without fixing what the loop spends its time on | `PAYOUT_HEARTBEAT` drops the gate's idle wait to 0.4 s and bought **nothing** on the round it was written for, because the loop could not come round in under 3.5 s regardless. A gate cannot fire faster than the work between gate checks. Measure the loop body before tuning the thing that decides how often to run it. |
| Treating any deck-counter rise as a fresh deal | A payout partly covering the counter turned **65 into a confident 1**; the rise back to 65 read as a new round, restarted the coin ledger at 1000 mid-round, and cost tracking for the next 56 frames. One digit, one round. Require the rise to land at `DECK_BEFORE_DEAL` or above — a deal that leaves 65 is not a deal. |
| Grouping a replay by the `round_index` in the log | That index was decided live by whatever `RoundTracker` looked like on the day, so a fixed boundary bug stays fixed only for future captures. Re-derive it: the journal records frames, not conclusions. |
| Leaving the turn gate on its idle heartbeat during a payout | The payout panels cover the deck pile, so the gate's key reads `None` for the whole animation — one distinct key, so it fires **once** and then waits out 5 s. The face-up meld lives in that gap. A five-card group call was watched through a whole round and the only frame read during its payout showed the table entirely covered. Drop to `PAYOUT_HEARTBEAT` while `ledger.pending`. |
| Testing geometry by painting through the same helper you read back through | Self-consistent and blind: reversing the meld's direction and shifting the whole row both broke nothing. Assert against the measured fact — the anchor — not against the other helper. |

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
4. ~~Static recognition from one frame~~ — **done, and measured every run.**
   `vision/layout.py` trims the letterbox and holds every region as a fraction;
   `geometry.py` finds and splits rows and reads frame colour; `templates.py` names the
   holomem; `digits.py` reads numbers; `group_labels.py` + `roster_panel.py` produce the
   roster. `reader.TableReader.read()` is the one entry point and returns a `FrameReading`
   in which **every absent field carries a reason** — never a silent default.

   Current scores, from `scripts/check_vision.py` and the badge harvester:
   cards **18/19 holomem, 19/19 colours, 0 wrong**; digits **17/17**; badges **56 read,
   0 wrong**; rosters **12 read, 0 wrong**. Card art is 62/62 holomem, so all 1365
   possible four-group rosters are covered. All fifteen group badges are captured.

   **Verify any region with `scripts/check_layout.py`**, which draws them back onto a
   frame — it caught four misplaced boxes first time. A fraction cannot be checked by
   reading it.

   Nothing here may guess: `identify` refuses on a thin *margin*, not a low score, because
   an unknown holomem still produces a plausible best match.
   **Do not read the roster from the panel's portraits** — card art scores 1/17 against
   them. Read the four group *labels*; the count of non-placeholder cells is the
   cross-check, and it only catches a label misread as a **differently sized** group.
   A count that is not 3, 4 or 5 means the panel is occluded, which is a free detector.
   Card-art filenames drift, so punctuation is normalised away and anything still
   unrecognised is **reported, never fuzzy-matched**. Tell the user the filename.

5. **Watching a live round.** `scripts/capture.py` + `vision/capture.py` +
   `vision/journal.py` grab the monitor, read each frame, append a line to
   `data/games/*.jsonl` and drop the frame. Three rounds are logged.
   `reader.TableReader` is deliberately **stateless and per-frame**; `vision/accumulate.py`
   is the layer above that remembers, and keeping them apart is what stops the reader
   pretending to have observed what it only recalls.
   `scripts/replay.py` runs a logged round back through it — the counterpart to
   `check_vision.py`, which scores frames rather than histories.

   `RoundBelief.observe(reading)` folds frames in. What it gets right, it gets right for a
   measured reason:

   - **Coins → scoring events.** `CoinLedger` settles a coin change only when it forms a
     *legal payout* — one winner, one payer or three, equal shares, any shortfall explained
     by a payer at zero. Settling eagerly does not work: the four seats are not read on the
     same frame, so a payout arrives piecemeal. Measured over the two complete rounds,
     eager settling gave **8 unexplained events and neither round balancing**; balance-gating
     gave **21 events, every amount a legal payout-table entry, both rounds closing at exactly
     4000 + minted**.
   - **Amount → the shape that scored.** `shapes_paying()` inverts the payout table. A
     triple's scoring set carries 0 or 3 bonus copies and never 1; a group carries 0 or 1.
     That is what makes most amounts decode to one shape, and 930 uniquely.
   - **Conservation → the table's size, without reading it.**
     `table = deck_size − deck_remaining − hands − scored`, hands being at the limit but for
     one seat mid-turn. Exact through unreadable seats and unobserved turns, and it is what
     catches a misread discard row: a floor deeper than the table can hold is a segmentation
     error that is otherwise invisible. It **refuses** whenever the ledger's books do not
     balance, since a missed payout would inflate it by exactly the amount that hides the bug.
   - **Discards → a floor, not a history.** `DiscardFloor` merges by `max` per card per seat,
     which needs no alignment between overlapping views. The append-per-turn design it
     replaced needs to see every turn, and the deck counter fell one at a time on only
     **17/81 and 11/59** draws — about a fifth. Sequence alignment was the other option and
     it is ambiguous exactly where it matters: `watame, watame, gura` overlaps a later
     `watame, gura, polka` two ways.

   `Tracking` is the guard, and it names what is missing rather than returning a bare bool.
   `ready` deliberately excludes `table`, which is a declared lower bound — folding it in
   would hold `ready` false all round and teach a reader to ignore it.

   - **The meld → `scored` by name.** `layout.PAYOUT_MELD` + `reader.read_meld()`. During a
     payout the game draws the scoring cards face-up mid-table, and that is the *only* moment
     `scored` is observable. `RoundBelief` attaches a meld to a settled payout only when the
     shape read off the cards is one the amount permits — two channels sharing no machinery,
     template matching against a number on screen. Agreement narrows `shapes` to one and turns
     `scored` from a range into named cards; disagreement attaches nothing and does **not**
     overrule the amount, which is the better-checked side.

   **Why `scored` by name is the gate for the whole overlay.** `belief.observe` computes
   `seen = hand + table + scored`. A round where 25–30 cards have scored away and `scored` is
   empty leaves the belief thinking a quarter of the deck is still live — systematically
   optimistic about completing hands, and indistinguishable from good advice. So a
   `PublicState` cannot be built from vision until the melds are read; it is not a
   "less precise" state, it is a wrong one.

   **A meld is drawn toward the seat that called, is right-anchored, and grows leftward.** So a
   position is a fixed right edge plus a vertical band, and `layout.meld_box(seat, cards)`
   derives the rest. All four seats are measured, and every card is 0.0527 wide wherever it is
   drawn — only the position and the vertical band move.

   Two mistakes on the way here are worth keeping. A **single** box: the two captures that
   pinned it agreed to within a pixel, and the coins on both show the *same seat won each time*
   — two samples of one condition look exactly like two samples. And a **predicted** box:
   symmetry put `bottom` at y 0.470–0.597 against a true 0.526–0.665, wrong by half a card. Its
   first attempt named the holomem correctly three times over and returned blue/blue/None
   against a true blue/pink/pink, because a triple is three of one holomem and identification
   cannot notice a slice that has drifted. **Only the colours ever catch that.**

   **A meld is cut straight out of known geometry, never segmented.** `layout.meld_cards`
   gives one box per card, sliced leftward from the anchor; `read_meld` identifies each. There
   is nothing to detect, so nothing to merge with — and a misplaced cut shows up as a card that
   will not identify, which the guards already handle, rather than as a count that quietly
   comes out one short.

   That replaced `find_row`, and segmentation is what had been eating **every group call**.
   `find_row` splits on felt, and a meld longer than three cards reaches left into the deck
   pile and the decoy card list, which merge with it into one stretch. A payout also rains
   coins across the table, orange enough to defeat colour-based separation. So a four-card call
   read as the rightmost three of itself — correctly refused by `meld_shape` as an incomplete
   group — and the fourth card was never seen. Seen live at least twice.

   `read_meld` returns **every** candidate — four seats × three legal lengths — and deliberately
   does not choose. A shorter slice fits inside a longer meld, so a four-card group call also
   yields a three-card candidate. `accumulate` chooses with the ledger's winning seat and the
   amount, which is strictly more evidence than the pixels.

   Positions were collected by **playing, not screenshotting**. `reader.survey_cards` +
   `geometry.frame_spans` report where frame-coloured things sit in `TABLE_INTERIOR`, on every
   frame that is read — 54 ms against the ~114 frames a round produces. It first ran only while
   a coin change was unexplained and that was too narrow: 17 frames of 114, mostly furniture,
   and both bottom-seat calls missed. It reports **spans as text, never a crop**: the game-over
   banner is drawn across the middle of the table and reads "&lt;player&gt;'s score hit 0", so
   a picture of that region is sometimes a picture of a username. Keep it until a four- and
   five-card meld have been read for seats other than `right`, then delete it along with
   `frame_spans` and `TABLE_INTERIOR`.

   `journal.Record.parse` reads old lines tolerantly — unknown fields dropped, renames
   migrated. The log *is* the history, so a rename must not shrink the evidence base.

   **A refused read is not the same as no information, and one round turned on that.** The
   bonus refused on all 86 frames: `amelia_watson` at 0.42 against `MIN_SCORE = 0.45`, on 68
   of them, the other 18 scattered one apiece across holomem *not even in the roster*. Pale
   blonde art on a pale ground gives cross-correlation little to grip — the same reason
   `hakui_koyori` tops out at 0.54 — so a correct read can sit under the floor all round, and
   without the bonus no hand can be priced. Three changes, none of them lowering the floor:

   - `templates.Match.best` records the top-ranked holomem even when refused, and
     `RoundBelief.bonus_by_vote` accepts it when the round agrees. **Not a lowered threshold —
     a larger body of evidence:** the bonus is the same card on every frame, so 86 reads are
     86 samples of one question. Requires `BONUS_VOTE_MIN` frames, a `BONUS_VOTE_RATIO` lead,
     and a holomem the roster can actually deal. The live case ran 68 against 3.
   - `identify(among=…)` ranks only against the roster. Measured: same answers, margins equal
     or better everywhere (`towa` +0.34→+0.49, `mio` +0.33→+0.42), the one genuinely bad crop
     still refused at +0.00. Nine of that round's twelve competing names were holomem the round
     could not deal.
   - The score-floor refusal **now reports the margin**. It did not, which is why the log
     could not distinguish a correct read under the floor from a coin-flip — in a module whose
     own docstring says the margin carries the decision. Measured separation on real cards:
     correct reads +0.22 to +0.49, non-cards +0.00 to +0.11.

   `MIN_SCORE` is still 0.45 and should stay there without evidence. The margin is what
   separates; the floor's remaining job is "this crop is not a card", and the margin does that
   too.

   The meld is drawn **in front of the bonus card**, so during a payout `BONUS_CARD` holds a
   meld card. What has been preventing a misread is the `expected=1` cross-check — about 1.8
   meld cards fall in that box, so the count disagrees and it refuses. Across three logged
   rounds `bonus` never once read as a different holomem. **Do not relax that `expected`.**

   Still unread: the two **side seats'** discards (diagonal staircases — `find_row` does not
   describe them and `read_discards` raises), **ranks**, and melds of **four or five cards**.

The browser panel already lives under that same constraint on purpose:
`Session._snapshot` feeds `Advisor.observe` on **every** state, not only the ones a
hint is asked about, because an unclaimed discard is visible only as a difference
between consecutive views. Moving that into `hint()` for speed would leave the
belief inferring from gaps and nothing would look broken — there is a test.

## Where the overlay may sit — it is not cosmetic

**The overlay must be inside the game's picture, clear of the black bars.** `layout.OVERLAY_SAFE`
is the checked zone: `x 0.020–0.360, y 0.020–0.165`, which on a 2880×1800 screen is
**979 × 234 px at (57, 122)** — over the menu button and the top opponent's card backs, neither
of which is read or worth looking at.

Getting this wrong does not fail loudly, or did not. A monitor grab composites whatever is on
screen and the overlay is always-on-top, so a panel in a **letterbox bar** is lit pixels outside
the picture — and `find_play_area` is a bounding box of lit pixels. At the shipped default of
(40, 40) it returned a picture 49 px too tall and 49 px too high, an aspect of 1.7235 against
1.7778, which the ±5% tolerance waved through. Regions then landed ~24 px out at mid-screen, so
reads came back **wrong rather than absent**: "found 6 glyphs, more than 5" on every coin box,
the roster unread on all 469 frames of a round, and no advice for two whole games.

`LETTERBOX_SKEW` now refuses it — a centred picture has equal bars, and across every capture on
disk they differ by at most **1 px** against the 49 px an overlay produces. `capture.py` prints
the reason rather than the usual "game not on screen".

The other hazard is the opposite one: a panel **inside a read region** is composited onto the
table and read as part of it, so the reader would be looking at whatever the advisor last said
— a feedback loop, not a misread. `OVERLAY_SAFE` is clear of all 37 read regions, with a 2%
margin from the picture's edges as well, since the panel is dark and covering the outermost rows
would shrink the detected area from the other direction.

The largest free rectangle is actually bottom-right, and it is **rejected on a ground geometry
cannot see**: the game draws its own Skip and Pokajan! buttons there, so a panel over them would
hide the controls the player needs even though clicks pass straight through.

## Running the overlay over a live game

Two commands, because pywebview owns a main loop and so does uvicorn:

```
.\.venv\Scripts\python scripts\capture.py --overlay      # watch, advise, serve
.\.venv\Scripts\python -m pokajan.server.overlay         # the window itself
```

`server/live.OverlayServer` runs the app on a daemon thread and publishes into `app.state.hub`,
so the screen reader becomes the producer `server/app.py` always said it would be — and
`web/overlay.js` cannot tell whether a `hint` message came from a browser game or from a
watcher pointed at the real one. Verified end to end against a real socket: **220 messages
published from a logged round, 220 received**, 170 carrying a recommendation and 50 explaining
why not.

Three rules the plumbing is built on:

- **The socket is one-way by construction.** `/overlay/ws` discards whatever the page sends and
  `OverlayServer` has no path back. Same reason the whole of M8 is read-only.
- **A dead overlay never disturbs the round.** A closed page, a server that would not start, a
  port already taken — all end with the watcher still reading and the log still being written.
  `publish` cannot raise.
- **Messages keep arriving even when there is nothing to say.** `overlay.js` fades a stale
  panel and renders `reason` when there is no hint, so a round that lost track explains itself
  instead of freezing on advice that is no longer true.

## What the overlay still cannot be told

`vision/state.public_view` produces a `PublicState` the untouched advisor consumes, so the
reader-to-advice chain is closed. Three fields still have no reader, and `PublicView.unknown`
names them rather than filling them silently:

- **`phase`** — draw and discard differ only by whether the seat holds eight cards, and only the
  viewer's own hand is countable.
- **`last_discard`** — the newest card in each field. This is the reader `data/pending/` crops
  were collected for, and it does not exist yet.
- **`turn_index`** — nothing counts turns independently. The deck counter cannot stand in: a
  claim does not move the turn, while a chained call draws several times without one.

`current_seat` *does* have a reader now (the four bars around the central oval), but the bars
are covered by every payout panel — which is exactly when the watcher samples hardest — so what
comes back is the last frame that could see them, with `PublicView.turn_read_at` saying when
that was. Read on 82 of 138 frames in the first live round to use it.

**That round confirmed the reader outright, including the `bottom` bar** (18 reads) which no
capture had ever shown lit. And it confirmed itself a second way, for free: collapsing the
readings into turns gives 28 of 37 transitions following `bottom → left → top → right` exactly,
and **every one of the other nine is a forward skip over a seat whose turn went unsampled —
not one goes backwards.** A reader producing false positives would scatter. Re-run that check
before trusting any change to `TURN_INDICATORS`; it needs no screenshots.

## Working with the user

They are the only source of game data, and they respond fast to **specific** requests —
name the exact file, value or screen. **Nothing is currently blocked on them** — every asset
the reader needs has been supplied. What is still worth having, in rough order of value:

- **A coin total or deck counter showing a 5.** Not blocking; see below.
- **A second frame of the zoomed-out wide shot** (only `20260809173544_1` has one). One frame
  is not enough to decide whether that camera position is a stable screen worth its own
  layout or a transient in the round-opening animation.
- **Rounds played with `scripts\capture.py` running.** No screenshots. Four-card group melds
  now read for both `right` and `top`, so right-anchoring is confirmed rather than assumed.
  **No five-card meld has ever been read**, and one specific failure is worth chasing: a
  bottom-seat Gen5 four-group read three of its four cards and was dropped, because
  `read_meld` requires every card. `read_meld` now records *why* a partly-read candidate was
  dropped, so the next occurrence says whether that is a score floor, a colour, or a box that
  is wrong for longer melds at that seat.

  **Do not "fix" it by inferring the missing member until that reason is known.** A group meld
  is one of each member, so four of five identified determines the fifth — a sound inference
  from the rules, and it would have rescued both cases. But if the cause is geometry rather
  than a weak match, that inference papers over a wrong box and writes a confidently wrong
  `scored`, which is the one failure this whole layer exists to prevent.

  The cause turned out not to be geometry at all: **the watcher polls at 0.45 Hz**, so a meld
  on screen for a second or two is sampled roughly once, if at all. A round with both a four-
  and a five-card call got exactly one mid-payout frame each and neither showed a card.

  That is now diagnosed and fixed, and the diagnosis is worth keeping because two rounds of
  confident guessing preceded it. The instrumented answer was **grab 71 ms, BGRA→RGB 0 ms** —
  so neither the capture nor the conversion, which is what both guesses had been. The 2 s a
  frame was `PendingStore`: PNG `optimize=True` plus a directory scan that ran twice per crop
  against a full 1221-file store. See the three rows in the measured-negatives table. A read
  frame during a payout should now cost about 0.77 s against the 4.4 s measured.

  **Confirmed on the next round**, which is the only reason it is written as fact. 150 records
  over 361 s against 32 over 152 s, and the gaps between `at` stamps went **min 3.50 → 0.55 s,
  p25 4.24 → 0.75 s, median 5.14 → 1.56 s**. Crops fell from 32 of 32 frames to 23 of 150. The
  p75 of 4.50 s is the idle heartbeat and is meant to be there. That round read 14 payouts with
  the books balancing and tracking clean.

  **The drop reason is now known, and it is a score floor, not geometry.** All 64 meld-slot
  refusals ever logged are score-floor refusals — no colour refusal, no failure that looks
  like a wrong box, and the 4- and 5-card slices refuse with furniture-like margins on frames
  where the 3-card meld read correctly, which is right-anchoring behaving. So the standing ban
  on inferring a group's missing member stays, but it is no longer what blocks anything.

  **A four-card group call has now been read, and confirmed twice over.** `top +390` came back
  as `4-group +1 bonus` from the coins and
  `[murasaki_shion:orange nakiri_ayame:blue yuzuki_choco:pink oozora_subaru:blue]` from the
  cards — four Gen2 members with exactly one copy of that round's bonus holomem,
  `murasaki_shion`. Two readers sharing no machinery agreeing on the shape *and* the bonus
  count. It was on screen for four consecutive frames, and the amount picked the 4-card
  candidate over the 3-card subset, which is what returning every candidate was for.

  **A five-group has now been caught, and it found a wrong box rather than a weak read.** The
  bottom seat took `+480` for a five-group — unambiguous, since that round's roster held no
  three-member group — and the log showed an ordinary bottom triple with *no meld refusal at
  all*. The three cards read were `tokino_sora, roboco, azki`: three of Gen0's five. The
  recorded survey then showed nothing frame-coloured left of 0.365 and content running past
  the old 0.523 anchor to 0.636, which rules out both right-anchoring and centring. The bottom
  seat's meld grows *rightward*; a triple is the same three boxes either way, so four rounds
  of bottom triples read perfectly while every longer bottom call was cut from felt.

  Fixed in `layout.MELD_GROWS_RIGHT`, with the silence fixed too — an extension now always
  records what it found.

  **Confirmed on the next round, and it is the strongest single result the reader has produced.**
  `bottom +1890 -- mono 5-group +1 bonus`, cards
  `[mori_calliope takanashi_kiara ninomae_inanis amelia_watson gawr_gura]`, every one blue. 1890
  inverts to exactly one cell with no competing shape; independently the cards are five distinct
  Myth members (complete group), all one colour (monochrome), and that round's bonus holomem
  `ninomae_inanis` appears exactly once (one bonus copy). Shape, colour and bonus count all
  agree with an amount read by a different reader off the coin displays. Zero refusals on that
  frame. The books closed at 4300 with 300 minted and `left` at 0, so the asymmetric floor and
  the coin-floor ending fired too. Recorded in `data/captures/payouts_observed.yaml`.

  So **five-card melds read, and the bottom seat's direction is confirmed by a read matched to
  its amount** rather than by absence in the survey. A wrong box cannot produce five distinct
  members of one group in the right colours. The four-card subset was also offered and correctly
  thrown out — four of five Myth members is not a complete group, which is why `read_meld`
  returns every candidate instead of choosing on pixels.

  **All four seats are now settled, and the meld geometry is closed.** `left` was the last, and
  it needed the sprint fix above to be caught at all: `left +930`, a monochrome four-group of
  Gen3 all in blue with that round's bonus `shirogane_noel` appearing once, and 930 inverts to
  that one cell. It **grows rightward**, like `bottom` and unlike `top` and `right`.

  The direction is legible from any long meld logged alongside its own three-card slice, with
  no pixels involved: growing rightward the slice is the meld's *first* three cards, growing
  leftward it is the *last* three. Across every meld ever logged that is 2/0 rightward for
  `left`, 3/0 for `bottom`, 0/19 for `top`, 0/16 for `right`, nothing contradicting.
  **Re-run that check before trusting any change to `MELD_ANCHORS`** — it is three lines of
  analysis over the journals and it needs no screenshots.

  Reclassifying `left` moves its anchor from the right edge of the three-card span to the left
  edge, 0.481 → 0.3229. The span itself is unchanged to four decimals, which matters because
  fourteen left triples had already been read correctly against the old value; there is a test
  pinning it. `MELD_UNSETTLED` is empty now, and the both-directions machinery stays: it is how
  a seat gets read while its direction is in doubt, at no measurable cost.

  They asked whether every position × hand-size combination was needed. It is not, and saying
  so saved roughly a dozen captures: the hand *size* is free from the payout amount, so only
  position was ever unknown — and right-anchoring then made even that one measurement per seat.
  Answer questions like this with the arithmetic rather than a count of screenshots.
- **A payout frame with the arrows and per-seat deltas legible**, when the split rule and the
  red/cyan delta colours come to be read. The deltas are drawn coloured, so
  `digits.MAX_INK_SATURATION` will have to be relaxed for them. Lower value than it was: the
  coin ledger already recovers every delta from the totals, and closed the books on both
  logged rounds without reading a single arrow.
- **A capture aimed at the two side seats' discards.** Now unblocked — `accumulate.py` exists
  and names those two seats in `DiscardFloor.unread` on every round. Note the design changed
  under this ask: the old plan of reading the newest card once per turn is dead (four turns in
  five pass unobserved), so what is wanted is a frame or two showing a **staircase at its
  fullest**, to work out how to segment a sheared row at all. Whatever is read from it merges
  as a lower bound, so a partial answer is still worth having.

Closed, and worth knowing why so none gets re-asked:

- **Card art: 62 of 62 holomem**, so all 1365 possible four-group rosters are fully covered.
  `scripts\check_vision.py` prints that combination count every run, which is the claim worth
  making — "complete for the rounds we have screenshots of" is not.
- **All fifteen group badges**, from seven "Groups coming up" reveal frames plus the table
  view. `scripts\harvest_group_labels.py` prints coverage every run.
- **The digit 5**, which appears in no number on the table — every payout is a multiple of
  ten, so no coin total ends in one. Cut from the Gen5 group *badge* instead, the badges
  being drawn in the same typeface (measured: coin-harvested exemplars match the Gen1–Gen4
  badges at 0.86–0.97). Still worth confirming from a real coin total if one ever shows a 5,
  since it is the one exemplar reasoned across contexts rather than read where it is used.

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
