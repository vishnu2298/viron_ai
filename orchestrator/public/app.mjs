// DOM wiring for the Viron Console. All model output is rendered with textContent (never innerHTML).
import { errorText, fmtMs, historyFrom, ladder, reasonText, summarizeStats, verdict } from './ui.mjs';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const KEY_STORE = 'viron.apiKey';
const store = {
  get() {
    try {
      return sessionStorage.getItem(KEY_STORE) || '';
    } catch {
      return '';
    }
  },
  set(v) {
    try {
      sessionStorage.setItem(KEY_STORE, v);
    } catch {
      /* storage blocked: key lives only for this page load */
    }
  },
};
let apiKey = store.get();
const messages = []; // { role, content, meta?, pending?, failed?, node }
let selected = null;

function headers() {
  const h = { 'content-type': 'application/json' };
  if (apiKey) h.authorization = `Bearer ${apiKey}`;
  return h;
}

async function api(path, init = {}) {
  const res = await fetch(path, { ...init, headers: { ...headers(), ...(init.headers || {}) } });
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error page */
  }
  return { ok: res.ok, status: res.status, body };
}

// ---------- header: health + stats ----------
async function refreshStatus() {
  const health = $('health');
  try {
    const res = await fetch('/health');
    health.className = `chip ${res.ok ? 'up' : 'down'}`;
    health.querySelector('.label').textContent = res.ok ? 'orchestrator up' : 'orchestrator down';
  } catch {
    health.className = 'chip down';
    health.querySelector('.label').textContent = 'orchestrator down';
  }
  try {
    const { ok, body } = await api('/api/v1/stats?hours=24');
    const s = ok ? summarizeStats(body?.data) : null;
    $('stat-total').textContent = s ? `${s.total} requests · ${s.windowHours}h` : 'stats unavailable';
    $('stat-tiny').textContent = s && s.tinyShare !== null ? `tiny ${s.tinyShare}%` : 'tiny —';
    $('stat-confident').textContent = s && s.confidentShare !== null ? `confident ${s.confidentShare}%` : 'confident —';
  } catch {
    $('stat-total').textContent = 'stats unavailable';
  }
}

// ---------- thread ----------
function render(m) {
  const node = m.node || el('div', `msg ${m.role}`);
  node.replaceChildren();
  node.className = `msg ${m.role}${m.failed ? ' failed' : ''}${m === selected ? ' selected' : ''}`;
  if (m.pending) {
    const b = el('div', 'bubble');
    const dots = el('span', 'typing');
    dots.append(el('span'), el('span'), el('span'));
    b.append(dots);
    node.append(b);
  } else {
    const bubble = el('div', 'bubble', m.content || (m.role === 'assistant' ? '(empty answer)' : ''));
    node.append(bubble);
    if (m.role === 'assistant' && m.meta) {
      const v = verdict(m.meta);
      const row = el('div', 'meta-row');
      row.append(el('span', `badge ${v.level}`, v.label), el('span', '', fmtMs(m.meta.latencyMs)));
      if (m.meta.contexts) row.append(el('span', '', `${m.meta.contexts} context`));
      node.append(row);
      bubble.tabIndex = 0;
      bubble.setAttribute('role', 'button');
      bubble.setAttribute('aria-label', 'Inspect this answer');
      bubble.onclick = () => select(m);
      bubble.onkeydown = (e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), select(m));
    }
  }
  if (!m.node) {
    m.node = node;
    $('thread').append(node);
  }
  node.scrollIntoView({ block: 'end', behavior: 'smooth' });
}

function select(m) {
  const prev = selected;
  selected = m;
  if (prev) render(prev);
  render(m);
  renderInspector(m.meta);
}

// ---------- inspector ----------
function renderInspector(meta) {
  $('inspector-empty').hidden = true;
  $('inspector-body').hidden = false;
  const v = verdict(meta);
  const box = $('verdict');
  box.className = `verdict ${v.level}`;
  box.replaceChildren(
    el('span', '', v.label),
    el(
      'small',
      '',
      v.level === 'confident'
        ? 'The serving tier passed the logprob confidence gate.'
        : 'No tier passed the confidence gate; this is the best local answer.',
    ),
  );

  const list = $('ladder');
  list.replaceChildren();
  for (const s of ladder(meta)) {
    const li = el('li', `step ${s.state}${s.served ? ' served' : ''}`);
    const head = el('div', 'step-head');
    head.append(el('span', 'step-name', s.tier));
    const badgeText = { confident: 'confident', unsure: 'unsure', error: 'error', skipped: s.onLadder ? 'not needed' : 'skipped' }[s.state];
    head.append(el('span', `badge ${s.state}`, s.served ? `${badgeText} · served` : badgeText));
    li.append(head);
    if (s.state === 'confident' || s.state === 'unsure') {
      li.append(el('div', 'step-detail', reasonText(s.reason)));
      const bar = el('div', 'bar');
      const fill = el('span');
      fill.style.width = `${Math.max(2, Math.min(100, s.tokenConfidencePct ?? 0))}%`;
      bar.append(fill);
      bar.title = 'Average per-token probability';
      li.append(bar);
      const nums = el('div', 'nums');
      nums.append(
        el('span', '', `p̄ ${s.tokenConfidencePct ?? '—'}%`),
        el('span', '', `mean lp ${s.meanLogprob ?? '—'}`),
        el('span', '', `low ${s.lowTokenRatio ?? '—'}`),
        el('span', '', fmtMs(s.ms)),
      );
      li.append(nums);
    } else if (s.state === 'error') {
      li.append(el('div', 'step-detail', s.error));
    }
    list.append(li);
  }

  const facts = $('facts');
  facts.replaceChildren();
  const add = (k, val) => facts.append(el('dt', '', k), el('dd', '', val));
  add('served by', meta?.servedBy ?? '—');
  add('router tier', meta?.tierRequested ?? '—');
  add('complexity', meta?.complexity ?? '—');
  add('adapters', meta?.adapters?.length ? meta.adapters.join(', ') : 'none');
  add('contexts', String(meta?.contexts ?? 0));
  add('latency', fmtMs(meta?.latencyMs));
  $('raw').textContent = JSON.stringify(meta, null, 2);
}

// ---------- send ----------
async function send(text) {
  const content = text.trim();
  if (!content) return;
  const history = historyFrom(messages);
  const user = { role: 'user', content };
  messages.push(user);
  $('empty')?.remove();
  render(user);
  const reply = { role: 'assistant', pending: true };
  messages.push(reply);
  render(reply);
  $('send').disabled = true;

  const tier = $('tier').value;
  const payload = { message: content, history, ...(tier ? { options: { tier } } : {}) };
  try {
    const { ok, status, body } = await api('/api/v1/chat', { method: 'POST', body: JSON.stringify(payload) });
    reply.pending = false;
    if (ok && body?.success) {
      reply.content = body.data?.answer ?? '';
      reply.meta = body.meta;
      select(reply);
    } else {
      reply.failed = true;
      reply.content = errorText(status, body);
      render(reply);
      if (status === 401) $('settings').showModal();
    }
  } catch {
    reply.pending = false;
    reply.failed = true;
    reply.content = errorText(0, null);
    render(reply);
  } finally {
    $('send').disabled = false;
    refreshStatus();
  }
}

// ---------- wiring ----------
const input = $('input');
const autosize = () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
};
input.addEventListener('input', autosize);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    $('composer').requestSubmit();
  }
});
$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  if ($('send').disabled) return;
  const text = input.value;
  input.value = '';
  autosize();
  send(text);
});
$('suggestions').addEventListener('click', (e) => {
  if (e.target instanceof HTMLButtonElement) send(e.target.textContent);
});
$('clear').addEventListener('click', () => window.location.reload());
$('open-settings').addEventListener('click', () => {
  $('api-key').value = apiKey;
  $('settings').showModal();
});
$('settings').addEventListener('close', () => {
  if ($('settings').returnValue === 'save') {
    apiKey = $('api-key').value.trim();
    store.set(apiKey);
    refreshStatus();
  }
});

refreshStatus();
setInterval(refreshStatus, 30_000);
input.focus();
