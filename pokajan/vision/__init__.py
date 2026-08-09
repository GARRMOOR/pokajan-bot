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
  tracking error.

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

  The panel crops portraits head-and-shoulders while cards show fuller art, so
  matching between them needs a crop step.

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
