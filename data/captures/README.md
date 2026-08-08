# Captured game data

Two kinds of thing land here, both fed by playing real rounds of Pokajan in
Hololive Dreams (milestone M0b). Both are needed before M2 can validate the rules
and long before M8 can recognise cards.

## 1. Payout observations — `payouts_observed.yaml`

Every time a Pokajan is called in a real game, record what happened. The goal is
to pin down the payout table, which is currently the largest unknown in
`rules/pokajan_v1.yaml`.

```yaml
- hand_type: triple          # triple | group
  group_size: null           # for group hands: how many members
  monochrome: false
  bonus_card: false
  claimed: true              # true = off a discard, false = self-draw/in-hand
  payer_lost: 120            # coins the paying player(s) each lost
  caller_gained: 120         # coins the caller gained
  note: "Suisei triple, mixed colours"
```

`payer_lost` and `caller_gained` are recorded separately on purpose — the
bankruptcy rule means they can legitimately differ, and confirming a case where
they do is one of the sharper tests of the engine.

The combinations worth prioritising, since they pin down the most config at once:
plain triple, monochrome triple, plain group, monochrome group, anything with the
bonus card, and one claimed-vs-self-drawn pair of the *same* hand (that pair alone
settles whether the payout source affects the amount).

## 2. Card art — `cards/`

Screenshots for template matching. Ideal is one clean crop per
`(character, colour)` variant, named `cards/<character_slug>_<colour>.png`.
Full-table screenshots are useful too — name them `tables/<timestamp>.png` — since
they capture real layout, scale, and lighting.

## Note on what gets committed

`.gitignore` excludes this directory's contents by default except `*.yaml`.
Screenshots of a live online game can contain other players' usernames, so they
stay local unless there's a reason to share them.
