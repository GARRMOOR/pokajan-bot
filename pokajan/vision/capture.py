"""Getting frames off the screen, and getting rid of them again.

Three separate jobs, kept apart because each has a different reason to exist.

**Grabbing.** `MonitorSource` hands back one frame at a time from a monitor. The game runs
borderless, so there is no window handle to chase and no Windows-specific dependency: grab
the monitor, let `layout.find_play_area` trim the letterbox, and refuse anything that is not
plausibly the game. Read-only, and that is the whole scope of M8 -- nothing here or anywhere
downstream sends input to the game.

**Deciding when a frame is worth reading.** A full read costs 285 ms and a grab 71 ms, both
measured live, so most polls must be cheap. `TurnGate` fires on the *deck counter's value*,
which costs 1.7 ms and is the game's own turn clock -- it falls by one on every draw.

Those two numbers are the whole poll budget and neither was the problem. A live round polled
once every 2.9 s and read once every 4.4 s, and the difference was `PendingStore` below:
compressing four crops and scanning a full directory twice per crop cost **2 s a frame**,
seven times the read it was delaying. Both are fixed, and crops are now rationed by
`crops_due`. The lesson is the one this file keeps relearning -- instrument the loop, do not
reason about it.

An earlier version waited for the table to hold still, on the reasoning that a settled table
is a safe one. It does not work and the measurement is worth keeping: over 45 seconds of real
play, **2 of 77 consecutive frame pairs looked settled**, the worst movement in *every* region
exceeded 190 of 255, and a round in which the deck fell from 53 to 20 produced six records.
Something on this table is always moving. `signature`, `moved` and `region_deltas` remain as
the instrument that produced that finding -- `scripts/capture.py --tune` -- not as the gate.

**Not keeping anything.** A frame never reaches the disk. It is grabbed into memory, read,
and dropped when the reference goes out of scope; there is no window in which a full
screenshot of a live online match exists as a file. That is stronger than deleting one
afterwards, and it is the point: those frames carry other players' usernames.

`PendingStore` is the single, deliberate exception, and it is temporary in both senses. Some
readers are not built yet -- the newest discard in each field, the turn gate, hand size from
card backs -- and their input cannot be reconstructed from the derived log, because the log
only holds what something already knows how to read. So a small **crop** of those regions is
kept, under a hard byte cap, oldest evicted first, purgeable with one command.

Crops are allowed by an explicit **allowlist of region names**, never by a blocklist. Two
reasons. A blocklist fails open: a region added later is saved by default, and the one that
would hurt is `COINS`, whose box deliberately reaches up over the player's name so that
`digits.split_digits` can find the number band beneath it. And the allowlist is checkable by
eye in one screenful, which a rule about what is forbidden is not.

**A hazard to design around before it arrives:** a monitor grab captures whatever is composited
onto the screen, and that will eventually include the advisor's own overlay window, which is
always-on-top by construction. An overlay sitting inside the play area would be read as part of
the table -- and it would be read as *whatever the advisor most recently said*, which is a
feedback loop rather than a misread. Two ways out, neither taken yet: keep the overlay outside
`find_play_area`'s bounds, or grab the game's window contents rather than the desktop. The
overlay's placement is already saved in `overlay.json`, so the first is checkable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol

import numpy as np

from . import layout

PENDING_DIR = Path(__file__).resolve().parents[2] / "data" / "pending"

# Cap on the crop store. 200 MB is a few thousand discard-field crops, which is far more
# than the readers still to be built need, and small enough not to matter on a full disk.
PENDING_MAX_BYTES = 200 * 1024 * 1024

# Regions a crop may be kept from. Every one shows cards, felt or the group panel and none
# of them contains text a person is identified by.
#
# COINS is absent on purpose and must stay absent: its box extends up over the player's name
# so that `digits.split_digits` can pick the number band out from underneath it, so a coin
# crop is a picture of somebody's username. RANKS is absent until it has been checked with
# `scripts/check_layout.py` rather than assumed.
CROPPABLE = frozenset({
    "deck", "deck_counter", "hand", "bonus_card",
    "group_panel", "group_labels", "reveal_panel", "reveal_labels",
    "discards_bottom", "discards_left", "discards_top", "discards_right",
})

# Regions whose contents decide whether the table has moved on. The flashing turn gate is
# deliberately not among them -- it changes twice a second on its own and would make every
# frame look new -- and neither is the bonus card, which carries a looping sparkle.
SIGNATURE_REGIONS: tuple[tuple[str, layout.Box], ...] = (
    ("deck_counter", layout.DECK_COUNTER),
    ("hand", layout.HAND),
    ("discards_bottom", layout.DISCARDS["bottom"]),
    ("discards_left", layout.DISCARDS["left"]),
    ("discards_top", layout.DISCARDS["top"]),
    ("discards_right", layout.DISCARDS["right"]),
)

# Signature detail: a 16x16 grid of mean brightness per region, compared with a tolerance.
#
# The tolerance is a threshold on the comparison, not a quantisation of the values, and the
# difference matters. Quantising first was tried and is subtly useless: it gives no tolerance
# at all near a bucket boundary, so brightening every pixel by one flips whichever cells
# happen to sit on an edge and the frame reads as changed. A threshold on the difference is
# uniform everywhere.
#
# 4 levels out of 256 absorbs re-render noise while still seeing one digit of the deck
# counter change -- the tightest real change on the table, since a draw may move nothing else
# visible when the drawer is an opponent. Both ends of that are pinned in the tests.
SIGNATURE_GRID = 16
SIGNATURE_TOLERANCE = 4
# Pixels are thrown away by striding before any arithmetic happens, down to about this many
# per axis. The pooled grid is 16x16, so a few samples per cell is plenty and the full-
# resolution passes were pure waste: measured, this took the signature from 26 ms to 2 ms per
# frame on a 2880x1800 grab. That is not micro-optimising -- Pokajan gives about ten seconds
# per turn, so a watcher that eats a quarter of a core is competing with the thing it advises.
SIGNATURE_SAMPLES = 64


@dataclass(frozen=True)
class Grab:
    """One frame, in memory only. Never written anywhere by anything here."""

    pixels: np.ndarray
    area: layout.PlayArea
    captured_at: float

    def crop(self, box: layout.Box) -> np.ndarray:
        return self.area.crop(self.pixels, box)


class CaptureError(RuntimeError):
    """The game could not be found on screen."""


# ------------------------------------------------------------------ grabbing ----
class Screen(Protocol):
    """The bit of `mss` this needs, so a test can stand in for a display."""

    @property
    def monitors(self) -> list[dict]: ...

    def grab(self, monitor: dict) -> object: ...

    def close(self) -> None: ...


class MonitorSource:
    """Frames from a monitor, with the game's play area found rather than configured.

    Which monitor is decided by grabbing each and keeping whichever one's contents pass
    `find_play_area` -- the 16:9 and letterbox checks. That is a real check and not a guess:
    it fails on a desktop, on a browser, and on the game mid-transition, so a bad answer
    announces itself instead of silently offsetting every region.

    **Not finding the game is never an error here.** It is the normal state most of the time:
    the game is fullscreen, so it is *not* on screen at the moment anyone types a command to
    start watching it. `grab` returns None and the caller keeps polling. An earlier version
    resolved the monitor once at startup and gave up if that failed, which made the watcher
    impossible to start at all -- you cannot launch it from a terminal that is covering the
    thing it is looking for.
    """

    def __init__(self, monitor: int | None = None, screen: Screen | None = None) -> None:
        if screen is None:
            import mss

            screen = mss.mss()
        self._screen = screen
        self._monitor = monitor
        self._chosen: int | None = monitor
        # Where a poll's time actually goes. Instrumented rather than reasoned about, because
        # the reasoning was wrong by a factor of thirty: this module's notes claimed "about
        # 40 ms of grab", and a live round measured **2.2 seconds between consecutive frames**
        # at best. That rate is what decides how often a payout meld -- on screen for a second
        # or two, and the only moment `scored` is observable -- gets seen at all.
        self.timing: dict[str, float] = {"grab": 0.0, "convert": 0.0, "grabs": 0.0}
        # Why the last grab was not a game picture. "Not on screen" is the ordinary reason and
        # needs no explanation, but "your overlay is sitting in the letterbox" is indispensable
        # and indistinguishable from it without this -- two rounds were watched and logged
        # entirely wrong before the check that produces this message existed.
        self.last_problem: str | None = None

    def close(self) -> None:
        self._screen.close()

    def __enter__(self) -> "MonitorSource":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    @property
    def monitors(self) -> list[dict]:
        # mss puts a virtual "all monitors" entry at index 0; the real ones start at 1.
        return self._screen.monitors[1:]

    @property
    def chosen(self) -> int | None:
        """Which monitor the game was last found on, or None if it has not been yet."""
        return self._chosen

    def _raw(self, index: int) -> np.ndarray:
        started = time.perf_counter()
        shot = self._screen.grab(self._screen.monitors[index])
        taken = time.perf_counter()
        # mss hands back BGRA. Drop alpha and reverse, rather than converting through PIL:
        # this runs on every grab and a megapixel round trip is not free.
        pixels = np.asarray(shot, dtype=np.uint8)[:, :, 2::-1]
        self.timing["grab"] += taken - started
        self.timing["convert"] += time.perf_counter() - taken
        self.timing["grabs"] += 1
        return pixels

    def find_game(self) -> int | None:
        """Which monitor is showing the game, or None if none is right now.

        Costs a frame, which is thrown away -- so this is for diagnostics. `grab` finds the
        monitor as part of its own work rather than calling this first, because doing both
        meant two full-screen grabs on every poll until one succeeded, at about 40 ms each.
        """
        return self._chosen if self.grab() is not None else None

    def describe(self) -> str:
        """What is actually on each monitor, for when the game is not being found.

        Worth having as a sentence rather than a shrug: the likely causes -- exclusive
        fullscreen capturing as black, a scaling mismatch, the wrong monitor -- all look
        identical from outside and quite different in these numbers.
        """
        lines = []
        for index in range(1, len(self._screen.monitors)):
            pixels = self._raw(index)
            height, width = pixels.shape[:2]
            area = layout.find_play_area(pixels)
            brightest = int(pixels.max())
            verdict = (f"play area {area.width}x{area.height} at ({area.x},{area.y})"
                       if area else
                       "no 16:9 letterboxed area"
                       + (" -- the grab is entirely black, which is what exclusive "
                          "fullscreen looks like; try borderless" if brightest <= 12 else ""))
            lines.append(f"  monitor {index}: {width}x{height}, brightest {brightest} -- "
                         f"{verdict}")
        return "\n".join(lines) or "  no monitors reported"

    def grab(self) -> Grab | None:
        """One frame, or None when the game is not currently on screen.

        None rather than an exception, because it is entirely ordinary: the player alt-tabs, a
        transition plays, the game has not been opened yet. A caller polling this keeps
        polling.
        """
        # Once a monitor is known, try only that one. `_chosen` is deliberately *not*
        # forgotten on a miss: the usual reason for a miss is that the game is not in front at
        # this instant, and rescanning every monitor on every poll to rediscover the same
        # answer would triple the cost of the common case on a multi-monitor desk.
        if self._chosen is not None:
            candidates: list[int] = [self._chosen]
        elif self._monitor is not None:
            candidates = [self._monitor]
        else:
            candidates = list(range(1, len(self._screen.monitors)))

        problem: str | None = None
        for index in candidates:
            pixels = self._raw(index)
            area, problem = layout._letterbox(pixels)
            if area is not None:
                self._chosen = index
                self.last_problem = None
                return Grab(pixels=pixels, area=area, captured_at=time.time())
        self.last_problem = problem
        return None


# -------------------------------------------------------------- change gate ----
def signature(grab: Grab,
              regions: tuple[tuple[str, layout.Box], ...] = SIGNATURE_REGIONS) -> np.ndarray:
    """A coarse fingerprint of the parts of the table that carry information.

    Coarse on purpose. The question is not "are these frames identical" -- they never are,
    once an animation loops somewhere -- but "has anything a reader would care about moved".
    Compare two of these with `moved`, never with `==`.
    """
    from PIL import Image

    blank = np.zeros((SIGNATURE_GRID, SIGNATURE_GRID), dtype=np.uint8)
    parts: list[np.ndarray] = []
    for _, box in regions:
        crop = grab.crop(box)
        if crop.size == 0:
            parts.append(blank)
            continue
        # Stride first, arithmetic second. The HAND box alone is 1630x350 and none of those
        # pixels survive a 16x16 pooling, so converting them all to float was the whole cost.
        step_y = max(1, crop.shape[0] // SIGNATURE_SAMPLES)
        step_x = max(1, crop.shape[1] // SIGNATURE_SAMPLES)
        sampled = crop[::step_y, ::step_x]
        # Image.BOX is exact area averaging, so this *is* the block-mean grid -- done in C
        # rather than as 256 numpy calls per region, because this runs on every grab.
        grey = Image.fromarray(sampled.astype(np.float32).mean(axis=2).astype(np.uint8))
        parts.append(np.asarray(grey.resize((SIGNATURE_GRID, SIGNATURE_GRID), Image.BOX)))
    return np.stack(parts) if parts else blank[None]


def moved(before: np.ndarray, after: np.ndarray) -> bool:
    """Whether two signatures differ by more than re-render noise."""
    if before.shape != after.shape:
        return True
    difference = np.abs(before.astype(np.int16) - after.astype(np.int16))
    return int(difference.max()) > SIGNATURE_TOLERANCE


def region_deltas(before: np.ndarray, after: np.ndarray,
                  regions: tuple[tuple[str, layout.Box], ...] = SIGNATURE_REGIONS
                  ) -> list[tuple[str, int]]:
    """Per-region worst change between two signatures, largest first.

    For calibration. `SIGNATURE_REGIONS` is a guess about which parts of the table hold still
    between turns, and one region that never holds still makes the whole table look
    permanently in motion -- so the gate stops firing and the log goes nearly empty. That
    happened: six records across a round in which the deck fell from 53 to 20. This is how the
    culprit gets named instead of guessed at.
    """
    if before.shape != after.shape:
        return [(name, 255) for name, _ in regions]
    difference = np.abs(before.astype(np.int16) - after.astype(np.int16))
    worst = [(name, int(difference[index].max()))
             for index, (name, _) in enumerate(regions)]
    return sorted(worst, key=lambda pair: -pair[1])


# How long the gate will sit without firing when the deck counter has not moved.
#
# Two values, because a payout is the one moment worth watching closely and the deck counter
# goes *blind* exactly then: the payout panels cover the pile, so the key reads None for the
# whole animation. That is a single distinct key, so the gate fires once and then waits out the
# idle heartbeat -- and the face-up meld, the only moment `scored` is observable at all, lives
# inside that gap. Measured: a five-card group call was watched through a round and the only
# frame read during its payout showed the table entirely covered.
#
# `TurnGate` applies `PAYOUT_HEARTBEAT` itself, the moment the counter refuses, rather than
# waiting to be told. Leaving it to the caller -- which dropped the heartbeat once
# `accumulate.CoinLedger.pending` reported an unexplained coin change -- reacts one signal too
# late, because noticing the coin change means having already sampled it. Measured, that
# version never sprinted at all: the gap to the next read while the counter was covered had a
# median of 2.46 s and a p90 of 5.30 s, which is this idle value rather than the one below.
#
# At a 0.3 s poll and about 250 ms of read that is roughly every other poll, for the few
# seconds a payout lasts, which is affordable in a way that polling that hard all round would
# not be: the counter refuses on about a quarter of frames.
IDLE_HEARTBEAT = 5.0
PAYOUT_HEARTBEAT = 0.4

# How often crops are worth keeping, and the answer is *far* less often than every frame.
#
# `PAYOUT_HEARTBEAT` above bought nothing on the round it was written for, and this is why. It
# drops the gate's idle wait to 0.4 s, but the loop could not come round in under 3.5 s: a read
# frame kept four crops, and those four crops cost 0.75 s of PNG compression plus 1.30 s of
# directory scanning -- **2 s against the 285 ms read they were delaying**. Both halves are
# fixed above; this is the third fix, and the one that would have mattered even without them.
#
# Crops feed a discard reader that does not exist yet, and that reader needs the newest card in
# each field about once a turn -- roughly ten seconds. Keeping them every read frame banked
# 1221 near-duplicates and filled the 200 MB cap, which is what made eviction expensive in the
# first place. So: at most one set per `CROP_INTERVAL`, and none at all while a payout is in
# flight, because that is the one moment the sampling rate is the whole product.
CROP_INTERVAL = 15.0


def crops_due(*, pending: bool, at: float, last: float,
              interval: float = CROP_INTERVAL) -> bool:
    """Whether this frame should spend time on crops, given a payout may be in flight.

    `pending` is `accumulate.CoinLedger.pending` -- a coin change seen and not yet explained,
    which is exactly when a face-up meld is on the table and `scored` is observable at all.
    Crops lose to that unconditionally, however long it has been since the last set.
    """
    return not pending and at - last >= interval


_UNSET = object()


@dataclass
class TurnGate:
    """Decides when a frame is worth the 150 ms a full read costs.

    **Not by waiting for the table to hold still. That was tried and it does not work.** The
    idea was that a settled table is a safe table, since a mid-animation frame is what produces
    a confident wrong answer. Measured over 45 seconds of real play: **2 of 77 consecutive frame
    pairs looked settled**, and the two that did were moments when every region was identical at
    once rather than gaps between turns. Something is always moving -- the felt, a card glow, a
    payout panel -- and the worst movement seen in *every* region was over 190 out of 255. A
    gate waiting for stillness fired six times in a round during which the deck fell from 53 to
    20, which is roughly one frame in four turns.

    So the signal is semantic instead of visual: **the deck counter's value**. It falls by one
    on every draw and a draw is what a turn is, so it is the game's own turn clock, and reading
    it costs 1.7 ms against 153 ms for a full read. `moved`/`signature` survive as the
    instrument that produced this finding -- see `scripts/capture.py --tune` -- not as the gate.

    A refusal counts as a distinct value, deliberately. When a payout display covers the
    counter the reading changes from a number to nothing, which fires the gate -- and a payout
    is the one moment a call's meld is observable at all, before those cards leave the table for
    good.

    `heartbeat` forces a read even when the key has not changed, so nothing depends on the
    counter alone: a round that ends on deck exhaustion stops moving it, and coins keep
    changing during a payout while it stays unreadable.

    **And a key that refuses is the payout signal itself, which is why `urgent` exists.** The
    counter is covered because the payout panels are over the pile, so a refusal is not a
    failure to interpret -- it is the one moment worth watching closely, arriving on the very
    frame it starts. Measured across every logged round, a meld is read on **14.4% of frames
    whose deck counter refused against 3.2% of frames where it read**.

    An earlier version left this to the caller, which dropped the heartbeat once
    `accumulate.CoinLedger.pending` said a coin change was outstanding. That is a strictly
    later signal and it showed: the gap to the next read *while the counter was covered* ran to
    a median of 2.46 s and a p90 of 5.30 s, which is the idle heartbeat, not the payout one.
    Seeing the coin change requires having already sampled it, and the meld and the coin change
    happen together -- so the sprint began, at best, after the thing it was meant to catch. A
    five-card left-seat group was lost in exactly that gap: one frame sampled inside its whole
    payout window, showing bare felt.
    """

    key: Callable[[Grab], object]
    heartbeat: float = IDLE_HEARTBEAT
    urgent_heartbeat: float = PAYOUT_HEARTBEAT
    # What a key looks like when the moment is worth watching closely. None by default means
    # "the deck counter refused", i.e. something is covering the pile. Injectable so a test can
    # say so without painting a payout panel.
    urgent: Callable[[object], bool] | None = None
    _key: object = _UNSET
    _fired_at: float = 0.0

    def offer(self, grab: Grab) -> bool:
        """True when this frame should be read. Advances the gate either way."""
        current = self.key(grab)
        pressing = self.urgent(current) if self.urgent else current is None
        wait = min(self.urgent_heartbeat, self.heartbeat) if pressing else self.heartbeat
        stale = grab.captured_at - self._fired_at >= wait
        if current != self._key or stale:
            self._key = current
            self._fired_at = grab.captured_at
            return True
        return False

    def forget(self) -> None:
        """Drop the memory of the last key, so the next frame is read whatever it shows.

        For round boundaries: a fresh deal can put the counter back to a value already seen,
        and the opening frame is the one frame the whole accumulation depends on.
        """
        self._key = _UNSET
        self._fired_at = 0.0


# ------------------------------------------------------------ the exception ----
@dataclass
class PendingStore:
    """The one place a picture is allowed to persist, and only for as long as it is needed.

    Crops, not frames, from an allowlist of regions that contain no usernames, under a byte
    cap with oldest-first eviction. Every part of that sentence is a constraint the caller
    cannot opt out of: `keep` refuses a region that is not on the list rather than trusting
    it, because the failure it prevents is writing somebody's name to disk.

    This exists because three readers are unbuilt and their input is not recoverable from
    the derived log -- a log holds what something already knows how to read. It should be
    deleted, along with `CROPPABLE`, when they land.
    """

    directory: Path = PENDING_DIR
    max_bytes: int = PENDING_MAX_BYTES
    allowed: frozenset[str] = CROPPABLE

    def keep(self, region: str, crop: np.ndarray, *, stamp: str, note: str = "") -> Path:
        """Write one crop. Raises for a region not on the allowlist."""
        if region not in self.allowed:
            raise CaptureError(
                f"{region!r} is not on the crop allowlist, so it will not be written. "
                f"Add it to capture.CROPPABLE only after checking it holds no player name "
                f"-- COINS is the one that does, and it is excluded on purpose."
            )
        from PIL import Image

        self.directory.mkdir(parents=True, exist_ok=True)
        suffix = f"_{note}" if note else ""
        path = self.directory / f"{stamp}_{region}{suffix}.png"
        # Deliberately *not* `optimize=True`, which was here and is the wrong trade despite
        # disk being the scarce resource on this machine. Measured on real discard fields it
        # writes 9% fewer bytes for 171-247 ms per crop against 25-68 ms plain -- four crops
        # spending 0.75 s of a frame's budget. And the 9% buys no disk at all: `max_bytes` is
        # a hard cap that eviction enforces either way, so smaller files mean about 9% more
        # crops retained, not a smaller store. Time is what is actually scarce here.
        Image.fromarray(np.asarray(crop, dtype=np.uint8)).save(path)
        self._evict(spare=path)
        return path

    # ----------------------------------------------------------- housekeeping --
    def _entries(self) -> list[tuple[Path, int]]:
        """Every kept file with its size, oldest first, from **one** directory scan.

        One scan, because the obvious spelling of this class was quadratic in disguise and it
        showed up in the one thing the whole module exists to do. `keep` evicts, eviction
        needs a total, and `files()` and `total_bytes()` each walked the directory and
        `stat`ed every entry: measured 131 ms and 193 ms against a full 1221-file store, so
        the four crops kept per frame spent **1.30 s** deciding what to delete. A full read of
        the table costs 285 ms. `os.scandir` carries the size along with the name, so the
        sizes come free and the second walk is gone.
        """
        if not self.directory.is_dir():
            return []
        found: list[tuple[Path, int, float]] = []
        with os.scandir(self.directory) as scan:
            for entry in scan:
                if not entry.is_file():
                    continue
                info = entry.stat()
                found.append((Path(entry.path), info.st_size, info.st_mtime))
        found.sort(key=lambda item: (item[2], item[0].name))
        return [(path, size) for path, size, _ in found]

    def files(self) -> list[Path]:
        return [path for path, _ in self._entries()]

    def total_bytes(self) -> int:
        return sum(size for _, size in self._entries())

    def purge(self) -> tuple[int, int]:
        """Delete everything. Returns (files removed, bytes freed)."""
        removed = freed = 0
        for path, size in self._entries():
            path.unlink()
            freed += size
            removed += 1
        return removed, freed

    def _evict(self, *, spare: Path | None = None) -> None:
        """Delete oldest-first until under the cap, never touching `spare`.

        Sparing the crop just written matters: without it a cap smaller than one crop makes
        `keep` delete the very thing it was asked to save and return a path to nothing. The
        cap exists to bound *accumulation*, and a single crop that exceeds it is a
        misconfiguration to notice rather than a request to silently ignore.
        """
        entries = self._entries()
        total = sum(size for _, size in entries)
        for path, size in entries:
            if total <= self.max_bytes:
                return
            if path == spare:
                continue
            total -= size
            path.unlink()


# ------------------------------------------------------------------ the loop ----
def frames(source: MonitorSource, gate: TurnGate, *, interval: float = 0.3,
           until: Callable[[], bool] | None = None,
           sleep: Callable[[float], None] = time.sleep,
           on_presence: Callable[[bool], None] | None = None) -> Iterator[Grab]:
    """Settled, changed frames, one at a time, until `until` says stop.

    A generator so the caller owns what happens to each frame and, more to the point, owns
    when the reference goes away. Nothing here holds on to a frame after yielding it.

    Waits for the game rather than requiring it. The game is fullscreen, so it is not on
    screen when anyone starts this -- they are looking at a terminal. `on_presence` is called
    only when the answer *changes*, so a caller can say "found it" and "gone" without
    printing either several times a second.

    A poll costs 71 ms of grab plus 54 ms of `find_play_area` plus 2 ms for the gate's deck
    read, measured live, so the 0.3 s default spends roughly a third of one core. That
    matters: Pokajan gives about ten seconds a turn and the watcher must not compete with the
    game it is advising on.
    """
    present: bool | None = None
    while not (until and until()):
        grab = source.grab()
        if (grab is not None) != present:
            present = grab is not None
            if on_presence:
                on_presence(present)
        if grab is not None and gate.offer(grab):
            yield grab
        sleep(interval)
