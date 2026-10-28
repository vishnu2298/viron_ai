const test = require('node:test');
const assert = require('node:assert/strict');
const router = require('../src/services/router.service');
const confidence = require('../src/services/confidence.service');
const { createElasticService } = require('../src/services/elastic.service');
const { createApp } = require('../src/app');

const CONF = { minMeanLogprob: -0.9, lowTokenLogprob: -2.5, maxLowTokenRatio: 0.15 };
const sure = Array.from({ length: 8 }, () => ({ logprob: -0.1 }));
const unsure = Array.from({ length: 8 }, () => ({ logprob: -3 }));

function elastic(local, traces) {
  return createElasticService({
    router,
    confidence,
    retrieval: { retrieve: async () => [] },
    inference: { generate: async ({ tier }) => ({ text: `${tier}!`, logprobs: local[tier] }) },
    traceRepo: { save: async (t) => traces.push(t) },
    config: { adapters: [], confidence: CONF },
  });
}

test('trace.confident reflects whether the serving tier passed the gate', async () => {
  const traces = [];
  await elastic({ tiny: sure, core: sure }, traces).chat({ message: 'hi' });
  await elastic({ tiny: unsure, core: sure }, traces).chat({ message: 'hi' });
  await elastic({ tiny: unsure, core: unsure }, traces).chat({ message: 'hi' });
  await new Promise((r) => setImmediate(r));
  assert.deepEqual(
    traces.map((t) => [t.servedBy, t.confident]),
    [
      ['tiny', true],
      ['core', true],
      ['core', false],
    ],
  );
});

test('GET /stats: localResolutionRate = confident answers / total', async () => {
  const rows = [
    { _id: 'tiny', count: 6, confident: 5, avgLatencyMs: 40 },
    { _id: 'core', count: 3, confident: 1, avgLatencyMs: 300 },
    { _id: 'escalation', count: 1, avgLatencyMs: 900 }, // legacy rows without the field count as not confident
  ];
  let askedHours;
  const traceRepo = {
    stats: async (h) => {
      askedHours = h;
      return rows;
    },
  };
  const srv = createApp({ elastic: {}, traceRepo, config: { env: 'test', apiKey: '' } }).listen(0);
  try {
    const res = await fetch(`http://127.0.0.1:${srv.address().port}/api/v1/stats?hours=12`);
    const { data } = await res.json();
    assert.equal(askedHours, 12);
    assert.equal(data.total, 10);
    assert.equal(data.localResolutionRate, 0.6);
    assert.deepEqual(data.byTier, rows);
  } finally {
    srv.close();
  }
});

test('GET /stats: no traces -> null rate', async () => {
  const srv = createApp({ elastic: {}, traceRepo: { stats: async () => [] }, config: { env: 'test', apiKey: '' } }).listen(0);
  try {
    const { data } = await (await fetch(`http://127.0.0.1:${srv.address().port}/api/v1/stats`)).json();
    assert.equal(data.total, 0);
    assert.equal(data.localResolutionRate, null);
  } finally {
    srv.close();
  }
});
