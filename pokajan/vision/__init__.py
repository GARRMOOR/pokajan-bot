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
"""
