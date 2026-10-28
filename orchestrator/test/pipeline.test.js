const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const router = require('../src/services/router.service');
const confidence = require('../src/services/confidence.service');
const { createInferenceService } = require('../src/services/inference.service');
const { createElasticService } = require('../src/services/elastic.service');
const { createApp } = require('../src/app');
const adapters = require('../src/config/adapters.json');

// Fake llama.cpp server: returns answers with configurable confidence and records requests
function fakeLlm(meanLogprob) {
  const calls = [];
  const server = http.createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      calls.push(JSON.parse(body));
      res.setHeader('content-type', 'application/json');
      res.end(
        JSON.stringify({
          choices: [
            {
              message: { content: `answer@${meanLogprob}` },
              logprobs: { content: Array.from({ length: 12 }, () => ({ token: 'x', logprob: meanLogprob })) },
            },
          ],
          usage: { completion_tokens: 12 },
        }),
      );
    });
  });
  return new Promise((r) => server.listen(0, () => r({ server, calls, url: `http://127.0.0.1:${server.address().port}` })));
}

function baseConfig(tiny, core) {
  return {
    env: 'test',
    apiKey: '',
    adapters,
    tiers: { tiny: { url: tiny.url, maxTokens: 64 }, core: { url: core.url, maxTokens: 128 } },
    confidence: { minMeanLogprob: -0.9, lowTokenLogprob: -2.5, maxLowTokenRatio: 0.15 },
  };
}

function build(cfg) {
  const traces = [];
  const elastic = createElasticService({
    router,
    confidence,
    retrieval: { retrieve: async () => [{ id: 'k1', text: 'Refunds are processed in 7 days.' }] },
    inference: createInferenceService({ tiers: cfg.tiers, timeoutMs: 2000 }),
    traceRepo: { save: async (t) => traces.push(t) },
    config: cfg,
  });
  return { elastic, traces };
}

test('tiny tier answers confidently: no expansion, LoRA attached, PII redacted in trace', async () => {
  const tiny = await fakeLlm(-0.1),
    core = await fakeLlm(-0.1);
  const { elastic, traces } = build(baseConfig(tiny, core));
  const out = await elastic.chat({ message: 'Refund status for me@mail.com?' });
  assert.equal(out.meta.servedBy, 'tiny');
  assert.equal(core.calls.length, 0);
  assert.deepEqual(tiny.calls[0].lora, [{ id: 2, scale: 0.8 }]);
  const sent = tiny.calls[0].messages;
  assert.equal(sent[0].role, 'system');
  assert.match(sent[0].content, /Context:\n\[1\] Refunds are processed/);
  assert.deepEqual(sent.at(-1), { role: 'user', content: 'Refund status for me@mail.com?' });
  assert.equal(sent.filter((m) => m.role === 'system').length, 1);
  await new Promise((r) => setImmediate(r));
  assert.ok(!traces[0].message.includes('me@mail.com'));
  tiny.server.close();
  core.server.close();
});

test('tiny unsure -> expands to core', async () => {
  const tiny = await fakeLlm(-3),
    core = await fakeLlm(-0.2);
  const { elastic } = build(baseConfig(tiny, core));
  const out = await elastic.chat({ message: 'Refund window?' });
  assert.equal(out.meta.servedBy, 'core');
  assert.equal(out.meta.attempts.length, 2);
  tiny.server.close();
  core.server.close();
});

test('both local tiers unsure -> best local (core) answer, never an external call', async () => {
  const tiny = await fakeLlm(-3),
    core = await fakeLlm(-3);
  const { elastic, traces } = build(baseConfig(tiny, core));
  const out = await elastic.chat({ message: 'Refund window?' });
  assert.equal(out.meta.servedBy, 'core');
  assert.equal(out.meta.escalated, false);
  assert.equal(out.answer, 'answer@-3');
  await new Promise((r) => setImmediate(r));
  assert.equal(traces[0].servedBy, 'core');
  tiny.server.close();
  core.server.close();
});

test('tiny down -> core serves (graceful degradation)', async () => {
  const core = await fakeLlm(-0.1);
  const { elastic } = build(baseConfig({ url: 'http://127.0.0.1:1' }, core));
  const out = await elastic.chat({ message: 'hi' });
  assert.equal(out.meta.servedBy, 'core');
  assert.equal(out.meta.attempts[0].tier, 'tiny');
  assert.ok(out.meta.attempts[0].error);
  core.server.close();
});

test('HTTP API: all tiers down -> 503 NO_TIER_AVAILABLE', async () => {
  const cfg = baseConfig({ url: 'http://127.0.0.1:1' }, { url: 'http://127.0.0.1:1' });
  const { elastic } = build(cfg);
  const srv = createApp({ elastic, config: cfg }).listen(0);
  const res = await fetch(`http://127.0.0.1:${srv.address().port}/api/v1/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ message: 'hi', options: { allowEscalation: true } }), // legacy field is stripped, not rejected
  });
  assert.equal(res.status, 503);
  assert.deepEqual(await res.json(), { success: false, message: 'No tier could produce an answer', code: 'NO_TIER_AVAILABLE' });
  srv.close();
});

test('HTTP API: validation + happy path', async () => {
  const tiny = await fakeLlm(-0.1),
    core = await fakeLlm(-0.1);
  const cfg = baseConfig(tiny, core);
  const { elastic } = build(cfg);
  const srv = createApp({ elastic, config: cfg }).listen(0);
  const base = `http://127.0.0.1:${srv.address().port}`;
  const bad = await fetch(`${base}/api/v1/chat`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' });
  assert.equal(bad.status, 400);
  const ok = await fetch(`${base}/api/v1/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ message: 'hello' }),
  });
  const json = await ok.json();
  assert.equal(json.success, true);
  assert.equal(json.meta.servedBy, 'tiny');
  srv.close();
  tiny.server.close();
  core.server.close();
});
