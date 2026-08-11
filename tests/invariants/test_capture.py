"""Grabbing frames, deciding which are worth reading, and keeping almost nothing.

No `mss` anywhere in here: a `Grab` is just an array plus a found play area, so every test
builds one directly. That is not a workaround -- it is the same reason the rest of the vision
suite is synthetic, and it means these run on a machine with no screen at all.

Two behaviours carry real weight and are pinned hardest.

**A frame is read when the deck counter's value changes**, because one draw is one turn. The
first design instead waited for the table to hold still, and that is a measured failure rather
than a refactor: across 45 seconds of real play only 2 of 77 consecutive frame pairs looked
settled, and a round in which the deck fell from 53 to 20 produced six records. The `signature`
tests below still earn their place -- that machinery is the tuning instrument that produced the
finding, reachable through `scripts/capture.py --tune`.

**A crop is written only from an allowlisted region.** The failure being prevented is writing
another player's username to disk, so the check has to fail closed -- an unknown region is
refused, never trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokajan.vision import layout
from pokajan.vision.capture import (
    CROPPABLE,
    SIGNATURE_REGIONS,
    SIGNATURE_TOLERANCE,
    CaptureError,
    Grab,
    PendingStore,
    TurnGate,
    frames,
    moved,
    signature,
)

pytestmark = pytest.mark.invariant

FELT = (34, 110, 34)


def grab(seed: int = 0, *, width=1920, height=1080) -> Grab:
    """A play area full of plausible table furniture, deterministic per seed."""
    rng = np.random.default_rng(seed)
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :] = FELT
    area = layout.PlayArea(x=0, y=0, width=width, height=height)
    for box in (layout.HAND, layout.DISCARDS["bottom"], layout.DISCARDS["left"],
                layout.DISCARDS["top"], layout.DISCARDS["right"]):
        left, top, right, bottom = box.pixels(area)
        pixels[top:bottom, left:right] = rng.integers(0, 255, (bottom - top, right - left, 3))
    return Grab(pixels=pixels, area=area, captured_at=float(seed))


def with_counter(base: Grab, digit_pattern: np.ndarray) -> Grab:
    """The same table with something different drawn in the deck counter."""
    pixels = base.pixels.copy()
    left, top, right, bottom = layout.DECK_COUNTER.pixels(base.area)
    patch = pixels[top:bottom, left:right]
    scaled = np.kron(digit_pattern, np.ones((patch.shape[0] // digit_pattern.shape[0] + 1,
                                            patch.shape[1] // digit_pattern.shape[1] + 1)))
    ink = scaled[:patch.shape[0], :patch.shape[1], None] > 0.5
    patch[:] = np.where(ink, np.uint8(235), np.uint8(60))
    return Grab(pixels=pixels, area=base.area, captured_at=base.captured_at)


# --------------------------------------------------------------- signatures ----

def test_the_same_table_has_not_moved():
    assert not moved(signature(grab(1)), signature(grab(1)))


def test_a_different_table_has_moved():
    assert moved(signature(grab(1)), signature(grab(2)))


def test_one_digit_of_the_deck_counter_is_enough_to_notice():
    """The tightest real change on the table, and the one that sets the tolerance.

    A draw moves the counter by one and may move nothing else visible -- an opponent's hand is
    card backs either way. If the signature cannot see that, the gate skips the frame and the
    accumulator misses a turn.
    """
    seven = np.array([[1, 1, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=float)
    one = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=float)
    base = grab(3)

    assert moved(signature(with_counter(base, seven)),
                 signature(with_counter(base, one)))


def test_a_tiny_change_everywhere_is_not_a_change():
    """The tolerance has to absorb re-render noise, or every frame looks new, nothing ever
    settles, and the gate reads *no* frames rather than too many.

    This is why the comparison is a threshold on the difference and not a quantisation of the
    values: quantising gives no tolerance at a bucket boundary, so brightening every pixel by
    one flips whichever cells happen to sit on an edge. Measured -- this test failed against
    the quantised version, which is what replaced it.
    """
    base = grab(4)
    nudged = Grab(pixels=np.clip(base.pixels.astype(np.int16) + SIGNATURE_TOLERANCE, 0, 255)
                  .astype(np.uint8), area=base.area, captured_at=0.0)

    assert not moved(signature(base), signature(nudged))


def test_the_signature_watches_the_regions_a_reader_cares_about():
    """It is a tuning instrument now, not the gate, but it still has to look at the right
    things: `--tune` reports movement per region, and a region missing from here cannot be
    diagnosed. The flashing turn indicator stays out because it changes on its own."""
    named = {name for name, _ in SIGNATURE_REGIONS}

    assert "gate" not in named and "bonus_card" not in named
    assert "deck_counter" in named and "hand" in named


# ---------------------------------------------------------------- turn gate ----
#
# The gate fires on the deck counter's VALUE, not on the table holding still. Waiting for
# stillness was tried and measured: 2 of 77 consecutive frame pairs looked settled across 45
# seconds of real play, and a round where the deck fell 53 to 20 produced six records. The
# `signature` tests above survive because that machinery is still the tuning instrument.

def counting_gate(values, **kwargs):
    """A TurnGate fed a fixed sequence of deck readings."""
    remaining = list(values)
    return TurnGate(key=lambda _: remaining.pop(0), **kwargs)


def at(seconds: float) -> Grab:
    base = grab(1)
    return Grab(pixels=base.pixels, area=base.area, captured_at=seconds)


def test_a_frame_is_read_when_the_deck_count_changes():
    """One draw is one turn, so the counter falling is the game's own turn clock."""
    gate = counting_gate([71, 71, 70, 70, 69], heartbeat=1e9)

    assert [gate.offer(at(t)) for t in range(5)] == [True, False, True, False, True]


def test_an_unchanged_count_is_not_read_again():
    """Otherwise every poll costs a 153 ms read of a table nobody has touched."""
    gate = counting_gate([46] * 6, heartbeat=1e9)

    assert [gate.offer(at(t)) for t in range(6)] == [True] + [False] * 5


def test_a_counter_that_cannot_be_read_is_its_own_state():
    """A payout display covers the pile, so the reading goes number -> None -> number. Each
    transition fires, and it must: a payout is the only moment a call's meld is observable
    before those cards leave the table for good."""
    gate = counting_gate([30, None, None, 30], heartbeat=1e9)

    assert [gate.offer(at(t)) for t in range(4)] == [True, True, False, True]


def test_the_heartbeat_reads_even_when_the_count_holds():
    """Nothing may depend on the counter alone. A round ending on deck exhaustion stops moving
    it at zero, and coins keep changing during a payout while it stays unreadable."""
    gate = counting_gate([0] * 4, heartbeat=5.0)

    assert [gate.offer(at(t)) for t in (0.0, 2.0, 4.9, 5.0)] == [True, False, False, True]


def test_forgetting_re_reads_whatever_comes_next():
    """For round boundaries: a fresh deal can put the counter back to a value already seen, and
    the opening frame is the one frame the whole accumulation depends on."""
    gate = counting_gate([71, 71], heartbeat=1e9)
    assert gate.offer(at(0)) is True

    gate.forget()

    assert gate.offer(at(1)) is True


# ------------------------------------------------------------------ the loop ----

def test_the_loop_yields_a_frame_per_change_and_survives_a_gap():
    class Source:
        def __init__(self, seeds):
            self._seeds = list(seeds)

        def grab(self):
            if not self._seeds:
                return None
            seed = self._seeds.pop(0)
            return None if seed is None else grab(seed)

    # A None in the middle is the player alt-tabbing: ordinary, and must not stop the loop.
    source = Source([1, 1, None, 2, 2, 2])
    counted: list[float] = []

    for frame in frames(source, TurnGate(key=lambda g: g.captured_at, heartbeat=1e9),
                        interval=0.0,
                        until=lambda: not source._seeds, sleep=lambda _: None):
        counted.append(frame.captured_at)

    assert counted == [1.0, 2.0]


# ----------------------------------------------------------- the crop store ----

def test_a_crop_is_kept_and_counted(tmp_path):
    store = PendingStore(directory=tmp_path, max_bytes=10 * 1024 * 1024)

    path = store.keep("discards_left", grab(1).crop(layout.DISCARDS["left"]), stamp="t0")

    assert path.exists() and path.parent == tmp_path
    assert store.total_bytes() == path.stat().st_size
    assert store.files() == [path]


def test_a_region_not_on_the_allowlist_is_refused(tmp_path):
    """Fails closed, and the region it protects is real: the COINS box reaches up over the
    player's name so `digits.split_digits` can find the number band beneath it, so a coin
    crop is a picture of somebody's username."""
    store = PendingStore(directory=tmp_path)

    with pytest.raises(CaptureError, match="allowlist"):
        store.keep("coins_left", grab(1).crop(layout.COINS["left"]), stamp="t0")

    assert store.files() == []


def test_the_coin_and_rank_regions_are_not_croppable():
    """Pinned as a list rather than left to the code, because adding a region is a one-line
    change and this is the line that would make it a mistake."""
    assert "coins_left" not in CROPPABLE
    assert not any(name.startswith("coins") for name in CROPPABLE)
    assert not any(name.startswith("rank") for name in CROPPABLE)


def test_the_cap_evicts_the_oldest_first(tmp_path):
    store = PendingStore(directory=tmp_path, max_bytes=1)
    crop = grab(1).crop(layout.DISCARDS["left"])

    first = store.keep("discards_left", crop, stamp="t0")
    second = store.keep("discards_left", crop, stamp="t1")

    assert not first.exists()
    assert second.exists()
    assert len(store.files()) == 1


def test_keeping_a_crop_never_deletes_that_crop(tmp_path):
    """A cap smaller than one crop must not make `keep` destroy what it was asked to save and
    hand back a path to nothing. The cap bounds *accumulation*; a single oversized crop is a
    misconfiguration to notice, not a request to silently ignore."""
    store = PendingStore(directory=tmp_path, max_bytes=1)

    path = store.keep("hand", grab(1).crop(layout.HAND), stamp="t0")

    assert path.exists() and path.stat().st_size > store.max_bytes


def test_purging_leaves_nothing(tmp_path):
    store = PendingStore(directory=tmp_path, max_bytes=10 * 1024 * 1024)
    for index in range(3):
        store.keep("hand", grab(index).crop(layout.HAND), stamp=f"t{index}")
    before = store.total_bytes()

    removed, freed = store.purge()

    assert removed == 3 and freed == before
    assert store.files() == [] and store.total_bytes() == 0


def test_a_missing_store_is_empty_rather_than_an_error(tmp_path):
    store = PendingStore(directory=tmp_path / "never-created")

    assert store.files() == [] and store.total_bytes() == 0
    assert store.purge() == (0, 0)


# ------------------------------------------------------- waiting for the game ----
class FakeScreen:
    """Stands in for mss: a list of monitors and whatever is currently on each.

    Frames are BGRA, because that is what mss hands back and the conversion is part of what
    is being tested -- getting the channel order wrong would turn every blue card orange.
    """

    def __init__(self, *, showing: list[bool]) -> None:
        self._showing = list(showing)
        self.grabs = 0

    @property
    def monitors(self) -> list[dict]:
        return [{"left": 0, "top": 0, "width": 1920, "height": 1080},      # the virtual one
                {"left": 0, "top": 0, "width": 1920, "height": 1080}]

    def grab(self, monitor: dict) -> np.ndarray:
        self.grabs += 1
        here = self._showing.pop(0) if self._showing else False
        if here:
            pixels = game_screen()
        else:
            pixels = np.zeros((1080, 1920, 3), dtype=np.uint8)             # a black desktop
        return np.concatenate([pixels[:, :, ::-1],
                               np.full((*pixels.shape[:2], 1), 255, np.uint8)], axis=2)

    def close(self) -> None:
        pass


def game_screen(width=1920, height=1080):
    """A monitor showing the game: 16:9 content inside letterbox bars."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    bar = (height - int(width * 9 / 16)) // 2
    pixels[bar:height - bar] = FELT
    return pixels


def test_the_game_not_being_on_screen_is_not_an_error():
    """The single most important thing here. The game is fullscreen, so it is *never* in front
    at the moment you type the command to start watching it -- you are looking at a terminal.

    An earlier version resolved the monitor once at startup and exited if that failed, which
    made the watcher impossible to start at all. Reported from real use, not imagined.
    """
    from pokajan.vision.capture import MonitorSource

    source = MonitorSource(screen=FakeScreen(showing=[False, False]))

    assert source.grab() is None
    assert source.chosen is None
    assert source.find_game() is None


def test_the_game_appearing_later_is_picked_up():
    from pokajan.vision.capture import MonitorSource

    source = MonitorSource(screen=FakeScreen(showing=[False, False, True]))

    assert source.grab() is None
    assert source.grab() is None
    grabbed = source.grab()

    assert grabbed is not None
    assert source.chosen == 1
    assert grabbed.area.aspect == pytest.approx(16 / 9, rel=0.05)


def test_finding_the_monitor_costs_one_grab_per_poll_not_two():
    """`grab` used to call `find_game` first, so every poll before the game appeared cost two
    full-screen grabs at about 40 ms each -- and each of those competes with a game that gives
    ten seconds a turn."""
    from pokajan.vision.capture import MonitorSource

    screen = FakeScreen(showing=[True, True, True])
    source = MonitorSource(screen=screen)

    source.grab()
    after_first = screen.grabs
    source.grab()

    assert after_first == 1
    assert screen.grabs == 2


def test_the_game_going_away_again_does_not_break_the_loop():
    """Alt-tabbing mid-round is ordinary and must not end the session."""
    from pokajan.vision.capture import MonitorSource

    source = MonitorSource(screen=FakeScreen(showing=[True, False, True]))

    assert source.grab() is not None
    assert source.grab() is None
    assert source.grab() is not None


def test_presence_is_reported_only_when_it_changes():
    """So a caller can say "found it" and "gone" without printing either three times a
    second for as long as the game is up."""
    from pokajan.vision.capture import MonitorSource

    source = MonitorSource(screen=FakeScreen(showing=[False, False, True, True, False]))
    changes: list[bool] = []
    remaining = [5]

    def tick(_):
        remaining[0] -= 1

    for _ in frames(source, TurnGate(key=lambda g: g.captured_at), interval=0.0,
                    until=lambda: remaining[0] <= 0, sleep=tick,
                    on_presence=changes.append):
        pass

    assert changes == [False, True, False]


def test_a_grab_comes_back_as_rgb_not_bgr():
    """mss hands back BGRA. Getting the order wrong would read every blue card as orange, and
    colour is a third of what identifies a card."""
    from pokajan.vision.capture import MonitorSource

    class Blue(FakeScreen):
        def grab(self, monitor):
            pixels = game_screen()
            pixels[pixels.any(axis=2)] = (10, 40, 200)          # decidedly blue, in RGB
            return np.concatenate(
                [pixels[:, :, ::-1], np.full((*pixels.shape[:2], 1), 255, np.uint8)], axis=2)

    grabbed = MonitorSource(screen=Blue(showing=[True])).grab()

    assert grabbed is not None
    middle = grabbed.pixels[grabbed.area.y + grabbed.area.height // 2,
                            grabbed.area.x + grabbed.area.width // 2]
    assert tuple(int(c) for c in middle) == (10, 40, 200)
