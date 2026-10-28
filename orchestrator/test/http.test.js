const test = require('node:test');
const assert = require('node:assert/strict');
const { createApp } = require('../src/app');
const { rateLimitKey } = require('../src/routes/v1');

const elastic = { chat: async () => ({ answer: 'ok', meta: { servedBy: 'tiny' } }) };

async function withApp(config, fn) {
  const srv = createApp({ elastic, config: { env: 'test', apiKey: '', ...config } }).listen(0);
  try {
    await fn(`http://127.0.0.1:${srv.address().port}`);
  } finally {
    srv.close();
  }
}

const chat = (base, headers = {}) =>
  fetch(`${base}/api/v1/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify({ message: 'hi' }),
  }).then((r) => Number(r.headers.get('ratelimit-remaining')));

test('rateLimitKey: CF header used only when trusted and a valid IP', () => {
  const req = (cf) => ({ ip: '10.0.0.1', headers: cf ? { 'cf-connecting-ip': cf } : {} });
  assert.equal(rateLimitKey({ trustCfIp: true })(req('203.0.113.7')), '203.0.113.7');
  assert.equal(rateLimitKey({ trustCfIp: true })(req('2001:db8::1')), '2001:db8::1');
  assert.equal(rateLimitKey({ trustCfIp: true })(req('not-an-ip')), '10.0.0.1');
  assert.equal(rateLimitKey({ trustCfIp: true })(req()), '10.0.0.1');
  assert.equal(rateLimitKey({ trustCfIp: false })(req('203.0.113.7')), '10.0.0.1');
  assert.equal(rateLimitKey()(req('203.0.113.7')), '10.0.0.1');
});

test('rate limit: TRUST_CF_IP gives each Cloudflare client its own bucket', () =>
  withApp({ trustCfIp: true }, async (base) => {
    assert.equal(await chat(base, { 'cf-connecting-ip': '203.0.113.1' }), 59);
    assert.equal(await chat(base, { 'cf-connecting-ip': '203.0.113.2' }), 59);
    assert.equal(await chat(base, { 'cf-connecting-ip': '203.0.113.1' }), 58);
  }));

test('rate limit: spoofed CF header is ignored unless trusted', () =>
  withApp({}, async (base) => {
    assert.equal(await chat(base, { 'cf-connecting-ip': '203.0.113.1' }), 59);
    assert.equal(await chat(base, { 'cf-connecting-ip': '203.0.113.2' }), 58);
  }));

test('trust proxy: X-Forwarded-For keys the limiter only when proxy is trusted', async () => {
  await withApp({ trustProxy: 'loopback' }, async (base) => {
    assert.equal(await chat(base, { 'x-forwarded-for': '198.51.100.1' }), 59);
    assert.equal(await chat(base, { 'x-forwarded-for': '198.51.100.2' }), 59);
  });
  await withApp({ trustProxy: false }, async (base) => {
    assert.equal(await chat(base, { 'x-forwarded-for': '198.51.100.1' }), 59);
    assert.equal(await chat(base, { 'x-forwarded-for': '198.51.100.2' }), 58);
  });
});
