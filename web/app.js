'use strict';

// The table UI. Its job is comparison, not polish: every payout is shown with the
// arithmetic that produced it, and the transcript reads like something you can
// hold next to a real round and check line by line.

let DATA = null;       // last payload from the server
let VIEW = null;       // history index being viewed, or null when live
let socket = null;

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- helpers ---

function slotChar(slot) { return Math.floor(slot / DATA.setup.colors.length); }
function slotColor(slot) { return slot % DATA.setup.colors.length; }
function colorName(slot) { return DATA.setup.colors[slotColor(slot)]; }
function charName(slot) { return DATA.setup.character_names[slotChar(slot)]; }

function groupOf(charIndex) {
  const groups = DATA.setup.group_members;
  for (let g = 0; g < groups.length; g++) {
    if (groups[g].includes(charIndex)) return DATA.setup.group_names[g];
  }
  return '';
}

function isBonus(slot) { return slotChar(slot) === DATA.setup.bonus_character; }

// Expand a count vector into a sorted list of slot indices, one per card held.
function expand(counts) {
  const out = [];
  counts.forEach((n, slot) => { for (let i = 0; i < n; i++) out.push(slot); });
  return out;
}

function cardEl(slot, { mini = false, onClick = null } = {}) {
  const el = document.createElement('div');
  el.className = `card ${colorName(slot)}${mini ? ' mini' : ''}${isBonus(slot) ? ' bonus' : ''}`;
  const who = document.createElement('div');
  who.className = 'who';
  who.textContent = charName(slot);
  el.appendChild(who);
  if (!mini) {
    const tag = document.createElement('div');
    tag.className = 'tag';
    tag.textContent = `${colorName(slot)} · ${groupOf(slotChar(slot))}`;
    el.appendChild(tag);
  }
  if (onClick) {
    el.classList.add('clickable');
    el.addEventListener('click', () => onClick(slot));
  }
  return el;
}

function send(msg) { socket.send(JSON.stringify(msg)); }

// ------------------------------------------------------------- rendering ---

function currentState() {
  if (VIEW !== null && DATA.history[VIEW]) return DATA.history[VIEW].state;
  return DATA.state;
}

function render() {
  if (!DATA) return;
  const live = VIEW === null;
  const state = currentState();

  $('meta').textContent =
    `${DATA.setup.rules_name} · ${DATA.setup.rules_hash.slice(0, 8)} · ` +
    `${DATA.setup.character_names.length} holomem · deck ${state.deck_remaining} left`;
  $('replay-banner').style.display = live ? 'none' : 'block';

  renderSeats(state, live);
  renderHand(state, live);
  renderTable(state);
  renderPrompt(state, live);
  renderPayouts();
  renderLog();
  renderQuestions();
  renderScrubber();
}

function renderSeats(state, live) {
  const host = $('seats');
  host.innerHTML = '';
  state.seats.forEach((s) => {
    const el = document.createElement('div');
    const isYou = s.seat === DATA.human_seat;
    const active = live && DATA.waiting_on.includes(s.seat);
    el.className = `seat${isYou ? ' you' : ''}${active ? ' active' : ''}`;

    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = isYou ? `Seat ${s.seat} — you` : `Seat ${s.seat}`;
    el.appendChild(name);

    const coins = document.createElement('div');
    coins.className = 'coins';
    coins.textContent = s.coins;
    // Colour the number by how close this seat is to ending the game.
    if (s.coins <= 200) coins.style.color = 'var(--bad)';
    else if (s.coins > DATA.setup.initial_coins) coins.style.color = 'var(--good)';
    el.appendChild(coins);

    const sub = document.createElement('div');
    sub.className = 'sub';
    sub.textContent = `${s.hand_size} cards · ${s.calls_made} pokajan · +${s.coins_won}/-${s.coins_paid}`;
    el.appendChild(sub);

    const recent = document.createElement('div');
    recent.className = 'recent';
    (state.recent_discards[s.seat] || []).slice(0, 6).forEach((slot) => {
      recent.appendChild(cardEl(slot, { mini: true }));
    });
    el.appendChild(recent);
    host.appendChild(el);
  });
}

function renderHand(state, live) {
  const host = $('hand');
  host.innerHTML = '';
  const decision = live ? DATA.decision : null;
  const canDiscard = decision && decision.decision === 'DISCARD';

  const cards = expand(state.hand);
  if (!cards.length) {
    host.innerHTML = '<span class="why">(empty)</span>';
    return;
  }
  cards.forEach((slot) => {
    const clickable = canDiscard && decision.legal_mask[slot];
    host.appendChild(cardEl(slot, {
      onClick: clickable ? (s) => send({ type: 'action', action: s }) : null,
    }));
  });
}

function renderTable(state) {
  const last = $('last-discard');
  last.innerHTML = '';
  if (state.last_discard_slot === null) {
    last.innerHTML = '<span class="why">nothing yet</span>';
  } else {
    const wrap = document.createElement('div');
    wrap.style.display = 'flex';
    wrap.style.alignItems = 'center';
    wrap.style.gap = '10px';
    wrap.appendChild(cardEl(state.last_discard_slot));
    const by = document.createElement('span');
    by.className = 'why';
    by.textContent = `from seat ${state.last_discard_seat}`;
    wrap.appendChild(by);
    last.appendChild(wrap);
  }

  const host = $('table-cards');
  host.innerHTML = '';
  const cards = expand(state.table);
  if (!cards.length) {
    host.innerHTML = '<span class="why">empty</span>';
    return;
  }
  cards.slice(-40).forEach((slot) => host.appendChild(cardEl(slot, { mini: true })));
}

function renderPrompt(state, live) {
  const host = $('prompt');
  host.innerHTML = '';
  host.className = 'prompt';

  if (state.finished) {
    host.classList.add('waiting');
    const order = state.final_coins
      .map((c, seat) => ({ c, seat }))
      .sort((a, b) => b.c - a.c);
    host.innerHTML =
      `<span class="what">Game over</span>` +
      `<span class="why">${order.map((o, i) =>
        `${i + 1}. seat ${o.seat} — ${o.c}${o.seat === DATA.human_seat ? ' (you)' : ''}`
      ).join(' · ')}</span>`;
    return;
  }

  if (!live) {
    host.classList.add('waiting');
    host.innerHTML = '<span class="why">Replay — press Live to resume playing.</span>';
    return;
  }

  const d = DATA.decision;
  if (!d) {
    host.classList.add('waiting');
    host.innerHTML = `<span class="why">Waiting on seat ${DATA.waiting_on.join(', ')}…</span>`;
    return;
  }

  const what = document.createElement('span');
  what.className = 'what';
  const why = document.createElement('span');
  why.className = 'why';

  const n = DATA.setup.n_slots;
  const CALL = n, PASS = n + 1;

  if (d.decision === 'DISCARD') {
    what.textContent = 'Discard a card';
    why.textContent = 'Click one in your hand. Anyone may claim it to complete a hand.';
    host.append(what, why);
    return;
  }

  if (d.decision === 'CLAIM') {
    what.textContent = `Claim seat ${currentState().last_discard_seat}'s discard?`;
    why.textContent = 'They alone pay. You keep your turn and owe no discard.';
  } else if (d.decision === 'IN_TURN_CALL') {
    what.textContent = 'Call Pokajan?';
    why.textContent = DATA.assumptions.discard_after_in_turn_call
      ? 'The other three split it. You will still discard afterwards.'
      : 'The other three split it.';
  } else {
    what.textContent = 'Chain another Pokajan?';
    why.textContent = 'Refilling draws from the shared deck — a long chain can end the game.';
  }
  host.append(what, why);

  if (d.best_call_payout !== null && d.best_call_payout !== undefined) {
    const chip = document.createElement('span');
    chip.className = 'payout-chip';
    chip.textContent = `${d.best_call_label} → ${d.best_call_payout} coins`;
    host.appendChild(chip);
  }

  if (d.legal_mask[CALL]) {
    const b = document.createElement('button');
    b.className = 'primary';
    b.textContent = d.decision === 'CLAIM' ? 'Claim' : 'Call Pokajan';
    b.onclick = () => send({ type: 'action', action: CALL });
    host.appendChild(b);
  }
  if (d.legal_mask[PASS]) {
    const b = document.createElement('button');
    b.textContent = 'Pass';
    b.onclick = () => send({ type: 'action', action: PASS });
    host.appendChild(b);
  }
}

function renderPayouts() {
  // Straight from the server's loaded config rather than hardcoded here, so this
  // panel can never drift from the numbers the engine is actually paying out.
  const t = DATA.payout_table;
  let html = '<table class="payouts"><tr><th>hand</th><th>multi-colour</th><th>single-colour</th></tr>';
  if (t) {
    html += `<tr><td>triple</td><td>${t.triple.multi}</td><td>${t.triple.mono}</td></tr>`;
    Object.keys(t.group).sort().forEach((size) => {
      html += `<tr><td>${size}-group</td><td>${t.group[size].multi}</td><td>${t.group[size].mono}</td></tr>`;
    });
  }
  html += '</table>';
  html += `<div class="why" style="margin-top:8px">Bonus holomem ★ ` +
          `<b>${DATA.setup.character_names[DATA.setup.bonus_character]}</b>: ` +
          `+${DATA.assumptions.bonus_per_copy} per copy the hand actually scores.</div>`;
  $('payouts').innerHTML = html;
}

function describeEvent(e) {
  const seat = (s) => (s === DATA.human_seat ? `you` : `seat ${s}`);
  switch (e.kind) {
    case 'deal':
      return ['muted', `Dealt. Bonus holomem is ${DATA.setup.character_names[e.bonus_character]}.`];
    case 'discard':
      return ['muted', `${seat(e.seat)} discarded ${e.card}`];
    case 'claim_passed':
      return ['muted', `nobody claimed ${e.card}`];
    case 'claim_won':
      return ['', `${seat(e.seat)} claimed ${e.card} from ${seat(e.from_seat)}` +
        (e.contenders.length > 1 ? ` (beat ${e.contenders.filter((c) => c !== e.seat).map(seat).join(', ')})` : '')];
    case 'pokajan':
      return ['pokajan', `POKAJAN — ${seat(e.seat)}: ${e.label}`];
    case 'payment': {
      const parts = e.payers.map((p) =>
        p.paid < p.owed ? `${seat(p.seat)} owed ${p.owed} but paid ${p.paid}` : `${seat(p.seat)} paid ${p.paid}`);
      const cls = e.minted > 0 ? 'minted' : 'payment';
      const tail = e.minted > 0 ? ` — ${e.minted} coins minted by the bankruptcy floor` : '';
      return [cls, `${seat(e.to_seat)} +${e.amount}: ${parts.join(', ')}${tail}`];
    }
    case 'refill':
      return ['muted', `${seat(e.seat)} refilled ${e.drawn} card(s), ${e.deck_remaining} left in deck`];
    case 'game_over':
      return ['over', `Game over — ${e.reason === 'deck_empty' ? 'deck exhausted' : 'a player hit zero'}. ` +
        `Final: ${e.coins.join(' / ')}`];
    default:
      return ['muted', e.kind];
  }
}

function renderLog() {
  const host = $('log');
  host.innerHTML = '';
  const upTo = VIEW === null ? DATA.history.length : VIEW + 1;
  DATA.history.slice(0, upTo).forEach((snap) => {
    snap.events.forEach((e) => {
      const [cls, text] = describeEvent(e);
      const el = document.createElement('div');
      el.className = `e ${cls}`;
      el.innerHTML = `<span class="turn">t${String(e.turn).padStart(2, '0')}</span> ${text}`;
      host.appendChild(el);
    });
  });
  host.scrollTop = host.scrollHeight;
}

function renderQuestions() {
  const host = $('questions');
  if (host.dataset.done) return;
  host.innerHTML = '';
  const intro = document.createElement('div');
  intro.className = 'q';
  intro.innerHTML = `<div class="c">These are the rules we are still assuming rather than
    knowing. Each one is a single line in <code>rules/pokajan_v1.yaml</code>. Check them
    against a real round and the guesses go away.</div>`;
  host.appendChild(intro);

  DATA.open_questions.forEach((q) => {
    const el = document.createElement('div');
    el.className = 'q';
    el.innerHTML =
      `<div class="t"><span class="impact ${q.impact}">${q.impact}</span>${q.text}</div>` +
      `<div class="c">${q.check}</div>`;
    host.appendChild(el);
  });
  host.dataset.done = '1';
}

function renderScrubber() {
  const slider = $('slider');
  const last = Math.max(0, DATA.history.length - 1);
  slider.max = last;
  if (VIEW === null) slider.value = last;
  $('pos').textContent = VIEW === null ? `live (${last + 1})` : `${VIEW + 1} / ${last + 1}`;
  $('live').disabled = VIEW === null;
}

// ------------------------------------------------------------------- wire ---

$('newgame').onclick = () => { VIEW = null; send({ type: 'newgame' }); };
$('live').onclick = () => { VIEW = null; render(); };
$('slider').oninput = (ev) => {
  const v = Number(ev.target.value);
  VIEW = v >= DATA.history.length - 1 ? null : v;
  render();
};
document.querySelectorAll('.tabs button').forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll('.tabs button').forEach((x) => x.classList.remove('on'));
    document.querySelectorAll('.pane').forEach((x) => x.classList.remove('on'));
    b.classList.add('on');
    $(`pane-${b.dataset.pane}`).classList.add('on');
  };
});

// Number keys discard the nth card, since clicking seven cards a turn gets old.
document.addEventListener('keydown', (ev) => {
  if (VIEW !== null || !DATA || !DATA.decision) return;
  const d = DATA.decision;
  if (d.decision === 'DISCARD' && ev.key >= '1' && ev.key <= '9') {
    const cards = expand(currentState().hand);
    const slot = cards[Number(ev.key) - 1];
    if (slot !== undefined && d.legal_mask[slot]) send({ type: 'action', action: slot });
  } else if (ev.key === 'c' && d.legal_mask[DATA.setup.n_slots]) {
    send({ type: 'action', action: DATA.setup.n_slots });
  } else if (ev.key === 'p' && d.legal_mask[DATA.setup.n_slots + 1]) {
    send({ type: 'action', action: DATA.setup.n_slots + 1 });
  }
});

function connect() {
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'error') { console.warn(msg.message); return; }
    DATA = msg;
    render();
  };
  socket.onclose = () => {
    $('prompt').className = 'prompt waiting';
    $('prompt').textContent = 'Disconnected — restart the server and reload.';
  };
}
connect();
