'use strict';

// The overlay's renderer. It receives finished advice and draws it — no game state,
// no card vectors, no rules. That is deliberate: everything it needs arrives as
// human-readable strings on the Recommendation, so when the screen reader replaces
// the simulator as the producer, nothing in here changes.
//
// It is also strictly one-way. The socket is never written to, and there is nothing
// to click. A renderer that could act would eventually be given a button, and then
// it is a second player in an online game against real people.

const PANEL = document.getElementById('panel');

// Placement mode is a query parameter rather than a separate page, so what you
// position is the same document you will be reading advice from.
const PLACE = new URLSearchParams(location.search).has('place');
if (PLACE) document.documentElement.classList.add('place');

// How long advice stays trustworthy with no word from the server. The real game
// allows about ten seconds a turn, so anything older than that is describing a
// position that has almost certainly moved on.
const STALE_AFTER_MS = 10_000;

let last = null;      // the last hint received
let lastAt = 0;

function esc(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function idle(text) {
  PANEL.className = 'panel';
  PANEL.innerHTML = `<div class="idle">${esc(text)}</div>`;
}

function draw() {
  if (!last || !last.hint) {
    // Placement usually happens with no game running, and an empty panel gives
    // nothing to judge the size against.
    idle(PLACE ? 'Advice appears here. Sized for two lines of reasoning.'
      : last && last.reason ? capitalise(last.reason) : 'Waiting for a decision…');
    return;
  }
  const h = last.hint;
  const stale = Date.now() - lastAt > STALE_AFTER_MS;
  const pct = Math.round(h.confidence * 100);
  // Same thresholds as the browser panel, and the same refusal to call it
  // confidence: it measures whether the ranking survives resampling the belief, not
  // whether the model is right.
  const colour = pct >= 80 ? '#4ade80' : pct >= 65 ? '#fbbf24' : '#ff6b6b';

  const alts = (h.alternatives || [])
    .map((a) => `${a.label} ${Math.round(a.behind) <= 0 ? 'level' : '-' + Math.round(a.behind)}`)
    .join('  ·  ');

  PANEL.className = `panel${stale ? ' stale' : ''}`;
  PANEL.innerHTML =
    `<div class="move">${esc(h.action_label)}</div>` +
    `<div class="row">` +
      `<span class="bar"><i style="width:${pct}%;background:${colour}"></i></span>` +
      `<span class="pct" style="color:${colour}">${pct}%</span>` +
      `<span class="tag">${stale ? 'STALE — position has moved on'
        : `stable to resampling · ${last.particles} samples · ${last.ms} ms`}</span>` +
    `</div>` +
    (h.reasoning ? `<div class="why">${esc(h.reasoning)}</div>` : '') +
    (alts ? `<div class="alts">next best — ${esc(alts)}</div>` : '');
}

function capitalise(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

// ------------------------------------------------------------------ resize ---
//
// A frameless window has no border for Windows to hand out resize edges from, so the
// corner grip does it instead, calling back into PlacementApi.
//
// Sized from clientX/clientY rather than from a screen-coordinate delta. clientX is
// measured from the window's own top-left, and the window is being resized under the
// cursor as we go, so it reads the new width directly and cannot drift — a delta
// against screenX would depend on whether screenX is CSS or device pixels, which is
// exactly the kind of unit question that made the geometry in overlay.py a mess.
//
// The only conversion left is CSS to physical for the call itself, and
// innerWidth * devicePixelRatio == the window's physical width exactly.
//
// This floor is about readability. overlay.py has a separate, smaller absolute floor
// so a runaway drag cannot shrink the window past its own grip.

const GRIP = document.getElementById('grip');
const MIN_CSS_WIDTH = 240, MIN_CSS_HEIGHT = 90;

if (PLACE && GRIP) {
  GRIP.addEventListener('pointerdown', (down) => {
    down.preventDefault();
    // Pointer capture, so the drag survives the cursor leaving the window — which it
    // does constantly, because the window edge is being pulled around under it.
    GRIP.setPointerCapture(down.pointerId);
    // Where inside the grip it was grabbed, so the corner does not jump to the
    // cursor on the first move.
    const holdX = window.innerWidth - down.clientX;
    const holdY = window.innerHeight - down.clientY;
    let want = null, queued = false;

    function flush() {
      queued = false;
      if (want && window.pywebview && window.pywebview.api) {
        window.pywebview.api.resize(want.w, want.h);
      }
    }

    function onMove(move) {
      const dpr = window.devicePixelRatio || 1;
      const cssW = Math.max(MIN_CSS_WIDTH, move.clientX + holdX);
      const cssH = Math.max(MIN_CSS_HEIGHT, move.clientY + holdY);
      want = { w: Math.round(cssW * dpr), h: Math.round(cssH * dpr) };
      // At most one resize per frame. A call per pointermove outruns the window
      // manager and the drag turns to treacle.
      if (!queued) { queued = true; requestAnimationFrame(flush); }
    }

    function onUp() {
      GRIP.releasePointerCapture(down.pointerId);
      GRIP.removeEventListener('pointermove', onMove);
      GRIP.removeEventListener('pointerup', onUp);
      flush();   // the last frame may still be pending
    }

    GRIP.addEventListener('pointermove', onMove);
    GRIP.addEventListener('pointerup', onUp);
  });
}

// Redrawn on a timer as well as on arrival, so the panel goes stale by itself. A
// feed that silently stopped would otherwise keep showing confident advice about a
// hand that finished minutes ago.
setInterval(draw, 1000);

function connect() {
  const socket = new WebSocket(`ws://${location.host}/overlay/ws`);
  socket.onmessage = (ev) => {
    last = JSON.parse(ev.data);
    lastAt = Date.now();
    draw();
  };
  socket.onclose = () => {
    last = null;
    idle('Table not running — start pokajan.server.app.');
    // Kept trying rather than dying, because the overlay outlives any one round and
    // there is nothing to click to restart it.
    setTimeout(connect, 2000);
  };
  socket.onerror = () => socket.close();
}
connect();
