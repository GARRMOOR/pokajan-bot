"""Reading the real game's screen (M8).

One thing is worth writing down now, long before this module has any code in it,
because it is the kind of detail that gets rediscovered expensively:

**Do not trust the game's "remaining cards" list.** It counts every card the
player has not *seen*, drawn from the full theoretical pool of 9 copies per
holomem — 153 cards for a 17-holomem lineup. The deck only ever holds 100. So
roughly a third of the entries in that list correspond to cards that are not in
the game at all, and the number it shows is not the number of cards left.

Scraping it and feeding it in as truth would give the agent a confidently wrong
picture of what is still live. The correct source for "what is left" is the
composition posterior in envs/belief.py, inferred from cards actually observed.
The counter is at best a weak upper bound.

This is also where the bot's structural edge comes from: a human reading that
list is being misled, and has no practical way not to be.

**Do not confuse that list with the number on the deck pile.** They are different
UI elements and only one of them lies. The pile shows the true count remaining out
of 100 -- observed at 71, 63, 46, 30 and 0 -- and should be read and trusted. The
card *list* is the decoy. Having conflated the two once already while reading
screenshots, it is written down here.

Exhaustion is unmistakable and does not need OCR to detect: the pile art disappears
entirely, leaving an empty outline, and the counter badge turns from dark green to
red. So the deck-empty end condition can be recognised by colour alone, which is
the right way round -- a misread digit there would have the reader thinking the game
continues past its end.

Facts established from real captures in data/tables/, all of which the reader
depends on:

* **Turn order is clockwise, which on screen means the player after you is the one
  on your LEFT** -- so seats run bottom, left, top, right. Note this is the
  opposite of the mahjong convention it otherwise resembles. Claim priority runs
  "in turn order starting from the player after the discarder", so mapping this
  backwards would resolve every tie to the wrong player, and would look like a
  strategy quirk rather than a bug.

* **Decisions are timed**, mahjong-style: roughly 10 seconds per turn plus a
  ~20 second reserve bank. Every agent here decides in well under a second, so the
  budget is not a constraint on the recommender -- but the overlay must render
  inside it.

* **A call displays its meld face-up in the centre**, with the hand type and amount
  spelled out ("3-Card 120"). So the cards a call consumed can be read at the
  moment it happens rather than reconstructed, which removes a whole class of
  tracking error. This is the *only* moment `scored` is observable -- those cards
  leave the table entirely afterwards -- so missing it means losing them for good.

  The payout display gives up considerably more than the meld, and all of it is
  printed rather than inferred:

  - **Each seat's coin delta and resulting total**, in its own panel: "-40 / 920",
    "+120 / 1380". So `coins` is read rather than accumulated by arithmetic, which
    means a missed event cannot compound into a wrong stack.
  - **Who paid whom**, as red arrows from each payer to the recipient, with the
    recipient's panel outlined in blue and the payers' in red. The split rule is
    therefore observable per event instead of assumed.
  - **The hand type by name**, so the reader never has to classify the meld shape
    itself -- and, better, every call becomes a free check of our own payout table
    against the game's answer. Worth wiring up as an assertion rather than a log
    line: a mismatch means `rules/pokajan_v1.yaml` is wrong.

  **But the panels occlude most of the table** -- the group panel, parts of the hand,
  and some discard fields sit underneath them. So "a payout is being displayed" is a
  state the reader must recognise and refuse to do a normal table read in, rather
  than something it can read through. The panels are large, rounded and near-white on
  green, so detecting them is easy; assuming they are not there is what would hurt.

* **Discards vanish from view two ways**: a claimed card leaves with the meld, and
  each seat's discard pile stacks once it grows, hiding the older cards. Neither
  removes them from the game. So `table` and `scored` must still be accumulated by
  watching continuously -- the reader cannot join a round in progress.

  The stacking was captured twice in one round, two minutes apart, and the geometry
  is worth having: a seat's discards run *away* from that seat, newest at the far
  end. Once the row is full the oldest cards slide underneath the card at the near
  end, offset a few pixels down-and-right, newest of the buried set on top. Between
  the two frames, two of the five cards in front of me became completely
  unreadable, and I only know what they were because I hold the earlier frame.

  Two consequences point in opposite directions and both matter:

  - The *recent* discards are fully readable from one frame, in order, because they
    are the un-stacked ones. So obs.py's per-opponent "last-3 discards" block is
    cheap and reliable even on a cold start.
  - The stack's depth reads as a handful of offset edges and then saturates, so a
    late-joining reader cannot recover the buried cards *or even their count*. That
    is a stronger statement than the claim-removal argument above: it means there
    is no clever frame to catch up from. Accumulate from the deal, and if tracking
    is lost, say so and stop advising rather than advise from a partial `table`.

  Together those settle what the reader should actually aim at, which is not what it
  looks like from a single screenshot. **Do not try to segment an opponent's discard
  field.** Measured: the seats either side of you lay their discards out as diagonal
  staircases, offset along the row as well as across it and sheared by the table's
  perspective, so the field's bounding box is far wider than one card and uniform
  slicing cuts across cards rather than between them -- aspects of 1.26 and 1.54 where
  a clean sideways card gives 1.395. Rectifying that is real work, and it buys the
  wrong thing: the old cards in the pile are already accumulated, and the buried ones
  are unreadable regardless. What a continuous reader needs is **the single newest
  card in each field, once per turn**, at the end furthest from its seat -- which is
  the un-stacked, unambiguous end.

* **Cards carry their group name** ("Gen1", "ID Gen3", "Myth") and colour is the
  card frame, not the artwork. So identification decomposes cleanly: character from
  an art template, colour from a flat frame sample, group as a free cross-check.
  One colourless template per character covers all three colours.

* **The roster is redrawn every round**, and the panel showing it **stays on the
  table for the whole round** -- it is not just an opening reveal. Confirmed at
  deck 30 and again at deck 0, unchanged. So there is no animation to catch and no
  timing dependency: the roster and the bonus holomem can be read from any frame,
  including the first one the reader ever sees. Group sizes observed: 4/4/3/5,
  4/4/4/3 and 4/4/4/5 (17 characters).

  The panel is a fixed 4x5 grid, and the grey right-pointing triangle in unused
  slots is a **placeholder, not a "more" affordance** -- nothing scrolls, and a
  group never exceeds 5. Reading group size is therefore counting non-placeholder
  cells, and the placeholder is a flat uniform grey that a mean-colour test
  separates from artwork without any character recognition at all.

  **The panel's portraits cannot be read with the card art, and they should not have
  to be.** Matching the panel's head-and-shoulders crops against card templates
  cropped to their head region scores 1/17 -- every cell lands around 0.2 to 0.36,
  far under the 0.45 the matcher requires, so a reader built that way refuses the
  whole roster. A crop step is not enough; the portraits are a genuinely different
  rendering.

  The way round it is not a second template set. **The four group labels are enough
  to determine the entire roster**, because the game draws real hololive branches and
  their membership is public knowledge rather than per-game. Every group size observed
  is a canonical one -- 4/4/3/5, 4/4/4/3 and 4/4/4/5, against GAMERS and Gen 3-5 at
  four, the ID generations at three, and Gen 0-2, Myth and holoX at five -- and the
  4/4/4/5 round was exactly GAMERS, Gen 4, Gen 5 and Myth.

  So reading the roster is: classify four short labels ("Ga", "4", "5", "My"), look
  their members up in a committed table, and cross-check the count against the
  non-placeholder cells in each row -- which is a flat-grey test needing no
  recognition at all. That replaces recognising up to twenty portraits with a
  four-way classification and an arithmetic check, and the check is what catches a
  misread label rather than trusting it.

  **This is built and measured.** Every capture whose panel is not covered reads all
  four badges at 0.94 to 0.98 with margins of 0.20 to 0.45, and produces the whole
  roster: three different rounds, 15, 16 and 17 characters, and the 17-character one
  matches the hand transcription in `data/captures/rounds_observed.yaml` exactly. No
  frame produced a wrong answer; the covered ones refuse. See vision/group_labels.py.

  Three things about the badges are not guessable from a single glance and each cost
  something to learn:

  - **A badge is not a digit, even when it looks like one.** Six groups print a bare
    numeral, but GAMERS prints "Ga", Myth "My", and the ID branches print their
    generation number with a small "ID" beneath: "1ID", "2ID", "3ID". Feeding a badge
    to the digit reader is not merely unhelpful, it is *confidently wrong* -- "Ga"
    reads as a 0 at 0.74 with its runner-up 0.25 behind, because a digit alphabet has
    nothing for a G to compete against and so nothing to refuse on.

  - **"1ID" and "1" are different groups**, and everything separating them is in the
    subscript. The first label box clipped it, so ID Gen1 read as a confident Gen1 --
    a real group, four different members, no complaint from anywhere. That is the
    whole failure mode of this project in one box edge: perception that answers
    instead of refusing.

  - **The badge is matched whole, with its aspect ratio intact** rather than stretched
    to fill a canvas. "1" is narrow and "My" is wide, and at eight exemplars the
    alphabet cannot afford to discard a dimension that cheap. Stretching also merges
    "1" into "1ID", which is the pair that matters most.

  Coverage is the live limitation: 8 of the 15 badges have been seen. The rest refuse,
  which is the intended behaviour but a weaker guarantee than digits enjoy, because an
  unseen two-glyph badge has seven seen ones to be wrong against. The member count is
  what catches that -- but only when the true and guessed groups differ in size, so
  Advent and Myth (both five) would not be separated by it. `LabelReader.uncovered`
  names what is missing.

  **The reveal screen is a trap worth knowing about.** Before the deal the game shows a
  "Groups coming up" screen presenting the same four rows much larger and elsewhere. The
  table boxes land on felt there, so the badges refuse -- but the panel box happens to
  count [5, 5, 5, 5], four perfectly legal group sizes, so the count check alone waves it
  through. The badges refusing is the only thing between that screen and a fabricated
  roster. Reading it properly needs its own layout and would be worth having: it is the
  cleanest, largest view of the roster the game ever shows.

* **The bonus holomem is displayed as a full card** beside that panel, under the
  word BONUS, all round. So it reuses the same templates as hands and discards
  rather than needing the portrait set, and it never has to be inferred from a
  payout. It also carries a sparkle effect that follows it into your hand, which is
  a second, redundant read on the same fact.

* **A seat's hand size is directly visible.** Card backs sit in one tight row, and
  a seat that has drawn and not yet discarded holds the drawn card **detached** from
  the row, mahjong-style. That is the visible form of hand size 8, and it is what
  makes the deck counter read one lower than a naive `100 - 28 - drawn` -- the
  reason the counter showed 71 rather than 72 in an earlier capture.

* **One of the four "gates" around the centre oval marks the seat that is to act --
  but it FLASHES rather than staying lit.** Confirmed. The flashing is the whole
  difficulty: a single frame catching the dark phase looks exactly like a frame
  where nobody is to act, which is what the two captures here disagreed about
  before this was known. So the gate must be read as a disjunction over a short run
  of frames -- lit in any of the last few means that seat is to act -- and never
  from one grab. A reader that samples once per decision and trusts the answer will
  report "no current seat" a large fraction of the time, and the failure looks like
  a flaky detector rather than a misread signal.

  Where a single frame *is* enough, prefer the detached card back above: it is
  static for as long as the seat owes a discard. The two are complementary rather
  than redundant, because the detached card cannot distinguish a claim window
  (where the seat to act has not drawn) from an ordinary turn.

* **Each seat's cards are drawn facing that seat, and the table has perspective.**
  The side seats' cards are not merely rotated 90 degrees, they are sheared into
  parallelograms. Template matching therefore needs a per-region affine
  rectification, not a rotation -- planning for rotation alone would produce a
  recogniser that works on my own hand and quietly fails on everyone else's.

* **Live rank badges (1st-4th) sit beside each seat's coins**, and a tie shows the
  same rank twice (observed: two seats both "1st" at 1430). Cheap to read, and it
  is exactly the quantity the safe agent is optimising against, so the overlay can
  display P(finishing above 1000) next to the game's own answer.

* **Coins totalled exactly 4000 in a game that ended on deck exhaustion** with the
  last seat still holding 10, against 4030 in the earlier game where a seat hit the
  floor. Minting is conditional on the floor actually biting, which is how the
  engine models it.

Unresolved, noted so it is not mistaken for something already understood: in the
deck-0 frame two cards in my hand -- both copies of the same character, different
colours -- carry a yellow border glow. Hover-highlighting every copy of the card
under the cursor is the obvious guess, but it could equally be a hint about a
scoring possibility, and the difference matters because the second reading would
mean the game is leaking advice the reader could just copy.
"""
