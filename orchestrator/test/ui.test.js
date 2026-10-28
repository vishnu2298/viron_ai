const test = require('node:test');
const assert = require('node:assert/strict');
const { createApp } = require('../src/app');

// ui.mjs is a browser ES module; its helpers are pure, so they run under node via dynamic import
const ui = import('../public/ui.mjs');

const meta = (over = {}) => ({
  servedBy: 'core',
  tierRequested: 'tiny',
  attempts: [
    { tier: 'tiny', confident: false, reason: 'low_mean_logprob', meanLogprob: -1.5, lowTokenRatio: 0.3, ms: 40, adapters: [] },
    { tier: 'core', confident: true, reason: 'ok', meanLogprob: -0.2, lowTokenRatio: 0.01, ms: 120, adapters: ['support'] },
  ],
  ...over,
});

test('ui: ladder marks each tier state and the serving tier', async () => {
  const { ladder } = await ui;
  const [tiny, core] = ladder(meta());
  assert.equal(tiny.state, 'unsure');
  assert.equal(tiny.served, false);
  assert.equal(core.state, 'confident');
  assert.equal(core.served, true);
  assert.equal(core.tokenConfidencePct, 81.9);
  const [t2, c2] = ladder(meta({ tierRequested: 'core', servedBy: 'core', attempts: [{ tier: 'core', error: 'down' }] }));
  assert.equal(t2.state, 'skipped');
  assert.equal(t2.onLadder, false);
  assert.equal(c2.state, 'error');
});

test('ui: verdict reflects the serving tier confidence', async () => {
  const { verdict } = await ui;
  assert.equal(verdict(meta()).level, 'confident');
  assert.equal(verdict(meta({ attempts: [{ tier: 'core', confident: false, reason: 'low_mean_logprob' }] })).level, 'unsure');
  assert.equal(verdict({ attempts: [] }).level, 'none');
});

test('ui: history, stats, formatting and error text', async () => {
  const { historyFrom, summarizeStats, fmtMs, errorText, tokenConfidencePct } = await ui;
  const msgs = [
    { role: 'user', content: 'a' },
    { role: 'assistant', content: 'b' },
    { role: 'assistant', content: 'x', failed: true },
    { role: 'assistant', pending: true },
    ...Array.from({ length: 30 }, (_, i) => ({ role: 'user', content: `m${i}` })),
  ];
  const h = historyFrom(msgs);
  assert.equal(h.length, 20);
  assert.deepEqual(h.at(-1), { role: 'user', content: 'm29' });
  assert.deepEqual(
    summarizeStats({
      total: 4,
      windowHours: 24,
      localResolutionRate: 0.5,
      byTier: [
        { _id: 'tiny', count: 3 },
        { _id: 'core', count: 1 },
      ],
    }),
    {
      total: 4,
      windowHours: 24,
      tiny: 3,
      core: 1,
      tinyShare: 75,
      confidentShare: 50,
    },
  );
  assert.equal(fmtMs(420), '420 ms');
  assert.equal(fmtMs(1234), '1.23 s');
  assert.equal(fmtMs(undefined), '—');
  assert.equal(tokenConfidencePct(null), null);
  assert.match(errorText(401), /API key/);
  assert.match(errorText(503, { code: 'NO_TIER_AVAILABLE' }), /No model tier/);
  assert.match(errorText(400, { details: ['"message" is required'] }), /message/);
});

test('GET /ui/ serves the console with CSP; API stays protected', async () => {
  const srv = createApp({ elastic: { chat: async () => ({}) }, config: { env: 'test', apiKey: 'k' } }).listen(0);
  const base = `http://127.0.0.1:${srv.address().port}`;
  try {
    const page = await fetch(`${base}/ui/`);
    assert.equal(page.status, 200);
    assert.match(page.headers.get('content-type'), /text\/html/);
    assert.match(page.headers.get('content-security-policy'), /script-src 'self'/);
    assert.match(await page.text(), /<title>Viron Console<\/title>/);
    const js = await fetch(`${base}/ui/app.mjs`);
    assert.equal(js.status, 200);
    assert.match(js.headers.get('content-type'), /javascript/);
    assert.equal((await fetch(`${base}/api/v1/stats`)).status, 401);
  } finally {
    srv.close();
  }
});
