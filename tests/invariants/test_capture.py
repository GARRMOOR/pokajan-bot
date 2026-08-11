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

from pokajan.vision import capture, layout
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
    gate = counting_gate([30, None, 30], heartbeat=1e9, urgent=lambda _: False)

    assert [gate.offer(at(t)) for t in range(3)] == [True, True, True]


def test_a_covered_counter_keeps_being_read_rather_than_waiting_out_the_idle_gap():
    """The counter is covered *because* a payout is on screen, so a refusal is the signal, not
    a failure. It arrives on the very frame the payout starts.

    Consecutive covered frames used to be one state change and then silence until the idle
    heartbeat — and the face-up meld lives inside that silence. Measured across every logged
    round: a meld is read on 14.4% of covered frames against 3.2% of readable ones, while the
    gap to the next read *while covered* had a median of 2.46 s and a p90 of 5.30 s. That is
    the idle heartbeat, in the one window that should be sprinting.
    """
    gate = counting_gate([None] * 4, heartbeat=5.0, urgent_heartbeat=0.4)

    assert [gate.offer(at(t)) for t in (0.0, 0.3, 0.4, 0.9)] == [True, False, True, True]


def test_urgency_can_only_ever_make_the_gate_fire_sooner():
    """`urgent_heartbeat` is a ceiling on the wait, not a replacement for it.

    Written plainly it is `urgent_heartbeat if pressing else heartbeat`, and then a caller who
    shortens `heartbeat` below it — which `scripts/capture.py` does, on `ledger.pending` — makes
    the *payout* frames the slow ones. Urgency that slows things down is worse than none, and it
    would show up as the meld window sampling less often the more certain we were about it.
    """
    gate = counting_gate([None] * 3, heartbeat=0.1, urgent_heartbeat=5.0)

    assert [gate.offer(at(t)) for t in (0.0, 0.05, 0.1)] == [True, False, True]


def test_the_sprint_is_decided_by_the_frame_in_hand_not_by_a_later_verdict():
    """This is the whole point, and the reason the previous attempt bought nothing.

    That version left the caller to drop the heartbeat once `CoinLedger.pending` reported an
    unexplained coin change. Noticing a coin change means having already sampled it, and the
    meld and the coin change happen together — so the sprint started, at best, after the thing
    it was meant to catch. A five-card left-seat group was lost exactly there: one frame
    sampled in its whole payout window, and it showed bare felt.

    So the gate must decide from the key it just computed, with nothing told to it.
    """
    gate = counting_gate([None, None], heartbeat=1e9, urgent_heartbeat=0.4)
    gate.offer(at(0.0))

    # No caller has touched `heartbeat`; it is still the idle value it was built with.
    assert gate.heartbeat == 1e9
    assert gate.offer(at(0.5)) is True


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


# ------------------------------------------------- the overlay's own pixels ----
#
# A monitor grab composites whatever is on screen, and the advisor's overlay is always-on-top
# by construction. Two rounds were watched and logged entirely wrong before these existed.

def screen(width=2880, height=1800):
    """A full screen with a centred 16:9 picture on it, letterbox bars and all."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    picture = int(round(width * 9 / 16))
    top = (height - picture) // 2
    frame[top:top + picture] = FELT
    return frame, top


PANEL = (18, 21, 29)          # web/overlay.html's .panel background, #12151d


def test_a_panel_in_the_letterbox_is_refused_rather_than_silently_shifting_everything():
    """The bug this whole check exists for, reproduced.

    With the overlay at its shipped default of (40, 40) the detected picture came back 49 px
    too tall and 49 px too high -- an aspect of 1.7235 against 16:9's 1.7778, which the +/-5%
    tolerance waves straight through. Regions then land about 24 px out at mid-screen, so reads
    do not fail, they come back *wrong*: "found 6 glyphs, more than 5" on every coin box, the
    roster unread on all 469 frames of a round, and advice that never arrived.
    """
    frame, top = screen()
    assert layout.find_play_area(frame) is not None, "the fixture must be a valid picture first"
    frame[40:40 + 190, 40:40 + 520] = PANEL

    assert layout.find_play_area(frame) is None
    why = layout.play_area_problem(frame)
    assert "letterbox" in why and "overlay" in why


def test_a_panel_inside_the_picture_changes_nothing():
    """Which is why the answer is to move it *in*, not out. Inside the picture the bounding box
    of lit pixels is already the picture, so the panel cannot move it."""
    frame, top = screen()
    before = layout.find_play_area(frame)
    box = layout.OVERLAY_SAFE
    left, top_y, right, bottom = box.pixels(before)
    frame[top_y:bottom, left:right] = PANEL

    assert layout.find_play_area(frame) == before


def test_the_safe_zone_touches_nothing_any_reader_reads():
    """Inside a read region the panel is composited over the table and read as part of it --
    which is worse than a misread, because the reader would be looking at whatever the advisor
    last said. A feedback loop, not an error."""
    box = layout.OVERLAY_SAFE
    regions = {name: getattr(layout, name) for name in
               ("DECK", "DECK_COUNTER", "GROUP_PANEL", "GROUP_LABELS", "REVEAL_PANEL",
                "REVEAL_LABELS", "BONUS_CARD", "HAND", "TABLE_INTERIOR")}
    for group in (layout.DISCARDS, layout.COINS, layout.RANKS, layout.TURN_INDICATORS):
        regions.update({f"{seat}{id(group)}": value for seat, value in group.items()})
    for seat in layout.SEAT_ORDER:
        for size in layout.MELD_SIZES:
            regions[f"meld{seat}{size}"] = layout.meld_box(seat, size)

    for name, region in regions.items():
        apart = (region.right <= box.left or region.left >= box.right
                 or region.bottom <= box.top or region.top >= box.bottom)
        assert apart, f"the overlay's safe zone overlaps {name}"

    # And clear of the picture's own edges, because the panel is dark: covering the outermost
    # rows would darken them and shrink the detected area from the other direction.
    assert box.left >= 0.02 and box.top >= 0.02
    assert box.right <= 0.98 and box.bottom <= 0.98


def test_a_centred_picture_survives_the_symmetry_check_at_any_size():
    """The check must not start refusing real screens. Measured across every capture on disk the
    two bars differ by at most 1 px, because letterboxing is integer arithmetic."""
    for width, height in ((2880, 1800), (1920, 1080), (3840, 2160), (1366, 768)):
        frame, _ = screen(width, height)
        assert layout.find_play_area(frame) is not None, f"{width}x{height} was refused"


def test_keeping_a_crop_walks_the_directory_once(tmp_path, monkeypatch):
    """The regression this guards cost more than everything it was measuring.

    `keep` evicts, eviction needs a total, and the natural spelling -- `files()` walking the
    directory and `total_bytes()` walking it again -- made that two full `stat` passes per
    crop. Against a full store that measured 131 ms and 193 ms, so the four crops a frame kept
    spent 1.30 s choosing what to delete while the read they delayed cost 285 ms. Counting the
    scans is the only way to see that from a test: it is invisible in the file listing, which
    is correct either way, and invisible on a `tmp_path` holding three files.
    """
    scans = []
    real = capture.os.scandir
    monkeypatch.setattr(capture.os, "scandir",
                        lambda path: scans.append(path) or real(path))
    store = PendingStore(directory=tmp_path, max_bytes=10 * 1024 * 1024)
    crop = grab(1).crop(layout.DISCARDS["left"])
    store.keep("discards_left", crop, stamp="t0")
    scans.clear()

    store.keep("discards_left", crop, stamp="t1")

    assert len(scans) == 1, f"{len(scans)} directory scans to keep one crop"


def test_a_crop_is_not_compressed_harder_than_the_read_it_delays(tmp_path, monkeypatch):
    """`optimize=True` cost 171-247 ms per discard crop against 25-68 ms plain -- four crops
    spending 0.75 s of a frame's budget -- and bought 9% fewer bytes, which buys no disk
    because `max_bytes` caps the store either way.

    Asserted on the call rather than the file size, having tried both. A timing assertion is
    flaky on a laptop, and a size assertion does not discriminate: these fixtures are random
    noise, which PNG cannot compress, so optimised and plain came out within 0.02% of each
    other and the test passed whatever the code did. The 9% is only there on real card art.
    """
    from PIL import Image

    saved: list[dict] = []
    real = Image.Image.save
    monkeypatch.setattr(Image.Image, "save",
                        lambda self, path, **kw: saved.append(kw) or real(self, path, **kw))
    store = PendingStore(directory=tmp_path, max_bytes=10 * 1024 * 1024)

    store.keep("discards_left", grab(1).crop(layout.DISCARDS["left"]), stamp="t0")

    assert saved and not saved[0].get("optimize")


# ------------------------------------------------------- when crops are due ----

def test_crops_are_never_kept_while_a_payout_is_in_flight():
    """However overdue they are. A face-up meld is on screen for a second or two and is the
    only moment `scored` is observable at all, so anything competing for that frame loses --
    and crops feed a reader that does not exist yet."""
    assert not capture.crops_due(pending=True, at=1_000.0, last=0.0)


def test_crops_are_kept_on_a_clock_rather_than_every_frame():
    """Every read frame was 1221 near-duplicates and a full 200 MB cap, which is what made
    eviction expensive. The unbuilt discard reader needs one sample a turn, about ten
    seconds."""
    interval = capture.CROP_INTERVAL

    assert capture.crops_due(pending=False, at=interval, last=0.0)
    assert not capture.crops_due(pending=False, at=interval - 0.01, last=0.0)
    assert capture.CROP_INTERVAL >= 10.0


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
