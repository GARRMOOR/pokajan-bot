"""The overlay window: advice pinned over the real game.

    python -m pokajan.server.overlay            # follows the table on :8000
    python -m pokajan.server.overlay --place    # movable, so you can position it
    python -m pokajan.server.overlay --check    # verify styles and placement, then exit

A frameless, transparent, always-on-top window that never takes focus and never
takes a click. That last pair is the whole reason this module exists rather than a
second browser tab: the real game is played online against real people, so an
advisory panel must be physically incapable of interfering with input. It reads
`Recommendation`s over a one-way socket and renders them, nothing else.

Three Windows details, each of which behaves differently from the obvious guess:

**Transparency is a colour key, not alpha.** pywebview's Windows backend implements
`transparent=True` by setting the form's `TransparencyKey` to pure red, so pixels
that end up exactly #ff0000 vanish. Semi-transparent panels therefore cannot work --
they composite against red and land on pink. `web/overlay.html` is built around this.

**Click-through needs `WS_EX_TRANSPARENT`, and the key colour gives it for free.**
Regions matching the transparency key already pass clicks through. The extended
style is what extends that to the opaque panel, so the whole window is inert. It
also means the window cannot be moved with the mouse, hence `--place`.

**Dragging goes through `.pywebview-drag-region`, not `easy_drag`.** easy_drag is
broken in pywebview 5.3.2: `util.py` lowercases the flag to `'true'` and
`customize.js` compares it against `'True'`, so it never binds and no argument here
could have made it work. The class-based path in the same file is fine. Two traps
around it: the listeners are attached by selector when the page loads, so the element
must exist in the markup and must not be one the renderer later rewrites; and
placement mode drops transparency, because key-coloured pixels are click-through at
the OS level and so cannot be grabbed at all. `--check` reports how many drag regions
the page actually presents.

**`focus=False` is not enough to stop it taking focus.** pywebview does apply
`WS_EX_NOACTIVATE` when asked, early in form construction, but reading the style back
afterwards shows it gone -- so something later in WebView2 setup clears it. Measured,
not assumed: `--check` reported `layered, click-through` and no `no-activate`. So this
module sets that flag itself, alongside the click-through ones. Losing it would mean
the overlay stealing the game's keyboard input at the worst possible moment.

**Where it sits is not cosmetic.** A monitor grab composites whatever is on screen and
this window is always-on-top, so a panel overlapping a letterbox bar is lit pixels
*outside* the game's picture -- and `layout.find_play_area` is a bounding box of lit
pixels. The shipped default of (40, 40) did exactly that: the reader measured the
picture 49 px too tall, every region landed about 24 px out, and two whole rounds were
logged with the roster unread and no advice given. `layout.OVERLAY_SAFE` is the zone
that is clear of both the bars and every region a reader touches, and `--check` reports
whether the saved geometry is inside it -- which only this process can do, since the
size is stored in CSS pixels and the zone is physical ones.

Position is remembered in `overlay.json` beside the repo (gitignored -- it is a
property of a monitor, not of the project), because a frameless click-through window
cannot be dragged and typing coordinates every round would be miserable.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from pathlib import Path

TITLE = "Pokajan overlay"
POSITION_FILE = Path(__file__).resolve().parents[2] / "overlay.json"

# Win32, from winuser.h.
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT = 0x00000020    # the window is skipped for hit-testing
WS_EX_LAYERED = 0x00080000        # required for WS_EX_TRANSPARENT to take effect
WS_EX_NOACTIVATE = 0x08000000     # never becomes the foreground window

# Applied together, after the window exists. pywebview's own focus=False handling
# does not survive WebView2 initialisation -- see the module docstring.
PINNED = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE

# Position in physical screen pixels, size in CSS pixels — and the mismatch is
# deliberate, because pywebview's geometry API is not self-consistent. Measured on a
# 200% display:
#
#   create_window(x=40)      -> window.x reads 40        (physical, round-trips)
#   window.move(120, 90)     -> window.x reads 240       (doubles; unusable)
#   create_window(width=520) -> window.width reads 988   (does NOT round-trip)
#   window.resize(700, 300)  -> window.width reads 700   (physical, round-trips)
#   innerWidth * devicePixelRatio == window.width        (exactly, both samples)
#
# So position is set at construction and never through move(), and size is set only
# through resize() after the window exists. Size is stored in CSS pixels because that
# is what governs whether the text is readable; storing physical pixels would make a
# panel saved on this laptop microscopic on a 100% monitor. The keys are named for
# their units so the next reader does not have to rediscover any of this.
DEFAULTS = {"x": 40, "y": 40, "css_width": 520, "css_height": 190}

# An absolute floor in physical pixels, purely so a runaway drag cannot shrink the
# window to nothing -- the grip would go with it and there is no chrome to get it
# back. The readability floor is a separate, larger one in CSS pixels in overlay.js.
MIN_WIDTH, MIN_HEIGHT = 200, 80


class PlacementApi:
    """Resizing, exposed to the page in placement mode only.

    A frameless window has no border for Windows to hand out resize edges from, so
    there is nothing to drag. Rather than restoring the frame -- which would move the
    content down by a title bar and make placement mode a different shape from the
    thing being placed -- the page draws its own grip and calls back here.

    **The underscore on `_window` is load-bearing.** pywebview walks this object's
    attributes to decide what to expose, recursing into any non-callable it finds,
    and it skips names beginning with an underscore. Holding the Window publicly
    makes it recurse into the Window, reach `window.dom.body`, and call `evaluate_js`
    before the window has started -- which fails the whole js_api injection with
    "Main window failed to start" and leaves the grip silently inert.
    """

    def __init__(self) -> None:
        self._window = None

    def resize(self, width: int, height: int) -> dict:
        clamped = (max(MIN_WIDTH, int(width)), max(MIN_HEIGHT, int(height)))
        if self._window is not None:
            self._window.resize(*clamped)
        return {"width": clamped[0], "height": clamped[1]}


# --------------------------------------------------------------- geometry ----
def load_position() -> dict:
    """Remembered geometry, falling back to defaults for anything missing.

    Tolerant of a corrupt or hand-edited file on purpose: a bad overlay position
    should put the panel in the wrong corner, not stop the tool from starting.
    """
    spot = dict(DEFAULTS)
    try:
        saved = json.loads(POSITION_FILE.read_text())
    except (OSError, ValueError):
        return spot
    for key in DEFAULTS:
        value = saved.get(key)
        if isinstance(value, int):
            spot[key] = value
    return spot


def physical_size(spot: dict, scale: float) -> tuple[int, int]:
    """The saved CSS size in the physical pixels `resize()` wants."""
    return (
        max(MIN_WIDTH, int(round(spot["css_width"] * scale))),
        max(MIN_HEIGHT, int(round(spot["css_height"] * scale))),
    )


def css_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """The inverse, for storing what the window reports back.

    Must round-trip with `physical_size` at any scale, or the window would grow or
    shrink a little on every launch -- which is precisely what the first version did,
    by saving physical pixels and passing them back as CSS ones.
    """
    return (int(round(width / scale)), int(round(height / scale)))


def save_position(spot: dict) -> None:
    try:
        POSITION_FILE.write_text(json.dumps(spot, indent=2) + "\n")
    except OSError as exc:
        print(f"could not save the overlay position: {exc}", file=sys.stderr)


# ------------------------------------------------------------------ win32 ----
def find_window(title: str, timeout: float = 15.0) -> int:
    """The window handle, once it exists.

    Found by title rather than through pywebview's `native` handle, which differs
    per backend. Polled because `webview.start` returns before the form is up.
    """
    user32 = ctypes.windll.user32
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hwnd = user32.FindWindowW(None, title)
        if hwnd:
            return hwnd
        time.sleep(0.05)
    return 0


def pin(hwnd: int) -> bool:
    """Make the window inert: no clicks, never activated. Returns whether it took.

    Read back rather than assumed, because that is how the missing no-activate flag
    was found in the first place.
    """
    user32 = ctypes.windll.user32
    before = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, before | PINNED)
    after = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    return all(after & bit for bit in (WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_NOACTIVATE))


def check_placement(spot: dict, width: int, height: int, scale: float) -> str:
    """Whether the window sits where the screen reader can still read the table.

    This is the check that should have existed first. The default position put the panel in the
    letterbox, which stretched the bounding box `layout.find_play_area` computes and shifted
    every region by about 24 px -- so reads came back *wrong* rather than absent, and two whole
    rounds were logged with the roster unread and no advice given.

    It lives here because this is the only process that knows the DPI scale, and the scale is
    the whole difficulty: size is stored in CSS pixels and the safe zone is physical ones, so
    the same `overlay.json` is fine at 100% and half a panel too big at 200%.
    """
    from ..vision import layout

    monitors = _screen_size()
    if monitors is None:
        return "cannot tell -- no screen size available"
    screen_w, screen_h = monitors
    picture_h = int(round(screen_w * 9 / 16))
    if picture_h > screen_h:
        return f"the screen is {screen_w}x{screen_h}, which is not 16:9 or wider -- not checked"
    top = (screen_h - picture_h) // 2

    box = layout.OVERLAY_SAFE
    x0 = math.ceil(box.left * screen_w)
    x1 = math.floor(box.right * screen_w)
    y0 = math.ceil(top + box.top * picture_h)
    y1 = math.floor(top + box.bottom * picture_h)

    x, y = spot["x"], spot["y"]
    if x >= x0 and y >= y0 and x + width <= x1 and y + height <= y1:
        return f"inside the safe zone (x {x0}-{x1}, y {y0}-{y1})"
    fits = f"{x1 - x0}x{y1 - y0}"
    return (f"OUTSIDE the safe zone x {x0}-{x1}, y {y0}-{y1}. The panel is {width}x{height} "
            f"physical at ({x},{y}). At {scale:g}x the largest that fits is "
            f"{int((x1 - x0) / scale)}x{int((y1 - y0) / scale)} css, at x {x0}, y {y0} "
            f"(the zone is {fits} physical). Anything in a letterbox bar makes the reader "
            f"mis-measure the picture and read every region off by tens of pixels.")


def _screen_size() -> tuple[int, int] | None:
    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return None


def check_handles(window, placing: bool) -> str:
    """Whether the page presents what placement mode needs to move and resize.

    Short of physically dragging the window, this is the check that matters. Both
    mechanisms fail silently: the drag listeners are attached by selector at page
    load, so an element that is missing, renamed, or created later is simply not
    draggable, and the resize grip is inert unless the js_api actually reached the
    page. The first of those is exactly how this shipped broken once already.
    """
    try:
        report = window.evaluate_js(
            "({drag: document.querySelectorAll('.pywebview-drag-region').length,"
            " grip: !!document.getElementById('grip'),"
            " api: !!(window.pywebview && window.pywebview.api"
            "         && window.pywebview.api.resize)})"
        )
    except Exception as exc:
        return f"could not check ({exc})"

    bad = []
    if not report.get("drag"):
        bad.append("no drag region — it will not move")
    if placing and not report.get("grip"):
        bad.append("no grip — it will not resize")
    if placing and not report.get("api"):
        bad.append("resize API did not reach the page")
    if bad:
        return "BROKEN: " + "; ".join(bad)
    return (f"{report['drag']} drag region(s)"
            + (", grip and resize API present" if placing else ", pinned so no grip"))


def describe_styles(hwnd: int) -> str:
    style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    flags = [
        name for name, bit in (
            ("layered", WS_EX_LAYERED),
            ("click-through", WS_EX_TRANSPARENT),
            ("no-activate", WS_EX_NOACTIVATE),
        ) if style & bit
    ]
    return f"0x{style & 0xFFFFFFFF:08x} [{', '.join(flags) or 'none of the three'}]"


# ------------------------------------------------------------------- main ----
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000,
                        help="port the table is served on")
    parser.add_argument("--place", action="store_true",
                        help="drag it anywhere to move, corner grip to resize, and it "
                             "saves where you left it. A pinned overlay can do neither, "
                             "because it takes no clicks at all")
    parser.add_argument("--check", action="store_true",
                        help="report the window styles that were applied, then exit")
    args = parser.parse_args(argv)

    if sys.platform != "win32":
        print("the overlay is Windows-only: it relies on Win32 window styles",
              file=sys.stderr)
        return 2

    try:
        import webview
    except ImportError:
        print("pywebview is missing — pip install -r requirements.txt", file=sys.stderr)
        return 2

    spot = load_position()
    api = PlacementApi()
    window = webview.create_window(
        TITLE,
        f"http://127.0.0.1:{args.port}/overlay" + ("?place=1" if args.place else ""),
        # Created hidden and at an arbitrary size. The real size is applied with
        # resize() once devicePixelRatio can be read, and only then is it shown -- so
        # there is no flash at the wrong size, and no moment where a not-yet-pinned
        # overlay is sitting on screen able to take a click or steal focus.
        hidden=True,
        width=DEFAULTS["css_width"], height=DEFAULTS["css_height"],
        x=spot["x"], y=spot["y"],
        frameless=True,
        # Opaque in placement mode. Key-coloured pixels are click-through at the OS
        # level, so a transparent window has nothing to grab except its panel; filling
        # it also shows the bounds you are actually positioning.
        transparent=not args.place,
        on_top=True,
        focus=args.place,
        resizable=args.place,
        # Only in placement mode. The pinned overlay must expose nothing callable:
        # it is a renderer, and a renderer with an API is one refactor away from
        # being a participant.
        js_api=api if args.place else None,
        # Deliberately not easy_drag. It is broken in pywebview 5.3.2 -- util.py
        # lowercases the flag to 'true' and customize.js tests it against 'True', so it
        # never binds. Dragging goes through the .pywebview-drag-region class in
        # web/overlay.html, which is the same file's other, working path.
        easy_drag=False,
    )

    # Recorded as the window moves rather than read back at the end. Querying
    # `window.x` after the loop exits throws from inside pywebview -- by then the form
    # is gone and `get_position` returns None -- which lost the placement every time.
    placed = dict(spot)
    scale = 1.0

    def remember_position(x, y) -> None:
        placed["x"], placed["y"] = int(x), int(y)

    def remember_size(width, height) -> None:
        placed["css_width"], placed["css_height"] = css_size(width, height, scale)

    if args.place:
        # Set directly, and privately: any public attribute here would be walked by
        # pywebview and exposed to the page. See PlacementApi.
        api._window = window
        window.events.moved += remember_position

    def configure(_window=None) -> None:
        # pywebview hands the window back as an argument; we already hold it.
        # --check must close the window on every path, including the ones that give
        # up early. A verification mode that hangs when the thing it verifies is
        # missing is the one case it exists to report.
        try:
            hwnd = find_window(TITLE)
            if not hwnd:
                print("could not find the overlay window; leaving it clickable",
                      file=sys.stderr)
                return
            if args.place:
                print("placement mode: drag it anywhere, then close it to save.")
            elif pin(hwnd):
                print("pinned: clicks pass through, and it will not take focus.")
            else:
                print("WARNING: pinning did not fully take. The overlay may swallow "
                      "clicks or steal focus — do not leave it over the game.",
                      file=sys.stderr)
            print(f"styles  {describe_styles(hwnd)}")
            # WebView2 initialises asynchronously. Nothing below can run until it has,
            # and tearing the form down mid-startup throws from inside the .NET
            # wrapper, which buries the report above in a stack trace.
            time.sleep(1.5)

            nonlocal scale
            try:
                scale = float(window.evaluate_js("devicePixelRatio")) or 1.0
            except Exception:
                scale = 1.0
            width, height = physical_size(spot, scale)
            window.resize(width, height)
            # Registered only now, so the conversion never runs against a stale scale.
            if args.place:
                window.events.resized += remember_size
            window.show()

            print(f"size    {spot['css_width']}x{spot['css_height']} css "
                  f"at {scale:g}x = {width}x{height} physical")
            print(f"placing {check_placement(spot, width, height, scale)}")
            print(f"handles {check_handles(window, args.place)}")
        finally:
            if args.check:
                window.destroy()

    webview.start(configure, window)

    if args.place:
        save_position(placed)
        print(f"saved {placed} to {POSITION_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
