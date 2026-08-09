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
of 100 -- observed at 71, 63 and 46 across a single round -- and should be read and
trusted. The card *list* is the decoy. Having conflated the two once already while
reading screenshots, it is written down here.

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

* **Cards carry their group name** ("Gen1", "ID Gen3", "Myth") and colour is the
  card frame, not the artwork. So identification decomposes cleanly: character from
  an art template, colour from a flat frame sample, group as a free cross-check.
  One colourless template per character covers all three colours.

* **The roster is redrawn every round** and is revealed up front on a "Groups
  coming up" panel, five slots wide with unused slots greyed. Group sizes observed:
  4/4/3/5 and 4/4/4/3. The panel crops portraits head-and-shoulders while cards
  show fuller art, so matching between them needs a crop step.
"""
