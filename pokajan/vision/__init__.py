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
  watching continuously -- the reader cannot join a round in progress. `accumulate.py`
  is where that lives, and it enforces the "cannot join" part rather than merely
  documenting it: a coin change it has no legal payout for is announced as a loss of
  track, which is exactly what a round joined halfway looks like from the inside.

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
  are unreadable regardless.

  That argument used to end "what a continuous reader needs is the single newest card
  in each field, once per turn, at the un-stacked end". **That is wrong, and the way
  it is wrong is worth keeping.** Once per turn requires seeing every turn, and over
  the two complete rounds in `data/games/` the deck counter fell one at a time on only
  17 of 81 draws and 11 of 59 -- about a fifth. Four turns in five pass without an
  individual observation, so an appended history would be missing most of the cards it
  claimed to hold and, worse, would not know which. The reasoning above was sound and
  its conclusion still did not survive contact with a stopwatch.

  What replaced it is in `accumulate.py` and needs no per-turn continuity at all. Every
  single view of a discard field is a true subset of what that seat has thrown, so the
  most of a card ever seen at once is a **lower bound**, merged by `max`. A bound that
  is short carries less information rather than false information, which is all the
  belief needs -- and the table's *size* comes from somewhere else entirely, by card
  conservation against the deck counter, exactly through the seats and turns this
  cannot see.

* **`geometry.CARD_ASPECT` is the aspect of a card in YOUR HAND, and does not
  transfer to the other seats.** Your hand sits closest to the camera; everything
  further up the table is foreshortened, so the same card is drawn shorter without
  being drawn narrower. Measured on live crops of lone cards: your own discards come
  out at 0.765 against the hand's 0.717, and the top seat's at 0.931.

  This mis-*counts* rather than merely mis-frames, because `find_row` derives the count
  from the aspect. The error is not symmetric, and that is why it hid for so long: the
  count is a rounded ratio, so the bottom seat's sub-7% error rounds away and its
  discards never looked broken, while the top seat's 30% error turned five cards into
  **seven** and drifted the boundaries far enough to read one card with its
  neighbour's colour. Fixed by `layout.DISCARD_ASPECT`, which `find_row` takes as a
  parameter. Card reading went from 11/12 to **18/19** once the top seat could be read
  at all.

  The two side seats have no entry on purpose: their discards are diagonal staircases
  rather than rows, so no single aspect describes them and `reader.read_discards`
  raises rather than returning plausible nonsense.

* **The top seat's cards are drawn upside down, and turning them upright reverses the
  row.** That seat's leftmost card is on the right of the screen, so slices come out in
  screen order and have to be flipped to be in *that seat's* order.

  Worth more care than it sounds, because discard order is information: obs.py encodes
  each opponent's last three discards, and the newest card sits at the end furthest
  from its seat. A reversed list puts the oldest card where the newest belongs and is
  wrong in precisely the way that looks right. Caught by reading a row as
  watame, watame, gura, polka off the screen and getting polka, gura, watame, watame
  back.

* **Identification confidence depends on the holomem, not just the size of the card.**
  Measured on live crops: the top seat's `shirakami_fubuki` scores 0.70 at 122x131,
  while the bottom seat's `hakui_koyori` tops out at 0.54 over an exhaustive crop
  sweep at 158x187 -- bigger, and worse. Pale pink art on a pale frame gives
  cross-correlation less to grip. So the 0.45 floor in templates.py is not a
  size threshold and must not be "fixed" by scaling it with the crop; the margin is
  what carries the decision, and on that card the margin was healthy at 0.28 once the
  framing was right.

  Two things that sound like the cause and are **not**, both checked: the query's
  aspect is already normalised away by `templates.query_variants`, which resizes every
  card to fixed dimensions before matching, so rectifying beforehand changes the score
  by nothing at all. And `FRAME_REFERENCES` transfers to live grabs unchanged --
  15 of 15 live cards classified correctly at distances of 10 to 33 against a limit of
  120 -- so the Steam-JPEG-versus-live-grab problem that broke the coin boxes does not
  extend to colour.

* **Cards carry their group name** ("Gen1", "ID Gen3", "Myth") and colour is the
  card frame, not the artwork.

  Live crops confirm this is worth reading rather than treating as decoration: a
  discarded card shows its badge plainly ("Gen5", "holoX", "ID Gen1"), which is a
  free cross-check of every discard against the roster that was read from the panel.
  A discard whose group is not one of the four dealt is a misread, and that check
  costs nothing because `group_labels` already classifies those badges.

* **The top seat's cards are drawn upside down**, not merely small -- they face that
  seat. `find_row(..., rotate=180)` is required there, and without it every template
  match is against an inverted portrait. So identification decomposes cleanly: character from
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

  **This is built, complete, and measured.** All fifteen badges the game can print are
  covered, on both screens that show them. Across every capture: **56 badges read, 0
  wrong, 12 refused; 12 rosters read, 0 wrong, 5 refused.** Scores run 0.82 to 0.98 and
  the tightest margin on a correct answer is 0.22, against a threshold of 0.15. Six
  distinct rosters of 15, 16 and 17 characters, and the 17-character one matches the hand
  transcription in `data/captures/rounds_observed.yaml` group for group. Every refusal is
  a frame where a payout panel covers the grid, or the reveal screen mid-animation.

  Four things about the badges are not guessable from a glance, and each cost something:

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

  - **The primary glyph and the subscript are matched separately.** Matching each badge
    as one picture worked at eight badges and broke at fifteen: "2ID" and "3ID" scored
    0.86 against each other, 0.14 of margin where the threshold wants 0.15, because the
    identical "ID" is most of the picture. Raising the canvas from 28px to 96px did not
    move it -- resolution does not change a ratio of shared to distinguishing ink.

    That is also the worst pair in the set to get wrong, because ID Gen2 and ID Gen3 both
    have three members, so the member count -- the one check independent of the reader --
    cannot break the tie. Compared as bare numerals they score 0.74, a margin of 0.26.
    The subscript is separable by geometry alone: a column run of its own, starting around
    0.55 of the badge's width and confined to the lower half, consistent across a fourfold
    scale change. It is then *matched* rather than merely counted, so a subscript that is
    present and does not look like "ID" refuses.

  - **Each glyph is stretched to fill the canvas, not letterboxed into it** -- the
    opposite of the intuitive choice, and measured. Preserving aspect is worse at every
    canvas size from 12x18 to 36x36: mean pairwise score 0.31 against 0.20, tightest real
    margin 0.15 against 0.27. Identical padding *correlates*, so two badges sharing
    nothing but their empty margins still agree over those margins and every score is
    dragged toward 1 together. Splitting the subscript off is what made stretching safe,
    since aspect was only ever separating "1" from "1ID".

  A badge is finally composed and then **checked against the list of badges the game
  actually prints**. Without that check a "Ga" with an "ID" under it reads as "GaID" at
  0.93, confidently, and there is no such group.

  **Both screens read, and neither is a fallback for the other.** The table panel sits out
  the whole round; the "Groups coming up" reveal screen before the deal shows the same four
  rows about four times the size, and is the largest and cleanest view of the roster the
  game ever gives -- worth catching because it arrives before the first turn. Each has its
  own boxes in layout.py. Pointed at the wrong screen, neither produces a single confident
  badge across every capture tried -- but *the table panel box counts [5, 5, 5, 5] on the
  reveal screen*, four perfectly legal group sizes, so the count check alone would wave it
  through. The badges refusing is the only thing between the wrong screen and a fabricated
  roster, which is worth knowing before anyone relaxes a threshold.

* **The bonus holomem is displayed as a full card** beside that panel, under the
  word BONUS, all round. So it reuses the same templates as hands and discards
  rather than needing the portrait set, and it never has to be inferred from a
  payout. It also carries a sparkle effect that follows it into your hand, which is
  a second, redundant read on the same fact.

* **A hand is not always a contiguous row, and this cost more than it should have.**
  A seat holds its drawn card detached, and a card leaving the hand leaves a hole
  until the row closes up. `geometry.find_row` measured from the first card to the
  last and divided by the card width, so a card-wide stretch of felt inflated the
  count and shifted every boundary -- slices landed on felt and refused. In one live
  session the same hand positions refused on **72 of 77 frames** while their
  neighbours read perfectly: positional, persistent, and nothing whatever to do with
  the art, which is where the search started. Each contiguous stretch is now sliced on
  its own and both diagnosable crops went from four refusals to **6 of 6 cards named**.

  A real hole and a gutter are nowhere near each other, which is what makes this safe
  next to the "do not split on felt" finding above: measured live, gutters run about
  0.05 of a card and a hole is 1.0. Nothing observed lands between.

  What the hole *means* is not settled. One crop shows five cards, a gap, then a lone
  card of a group that does not sort beside its neighbours -- a drawn card, plainly.
  Another shows four, a gap, then two, split exactly at a group boundary, which that
  story does not explain. `CardRow.detached` therefore reports structure only; reading
  it as "this seat has drawn" would turn an animation frame into a wrong turn
  attribution.

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
