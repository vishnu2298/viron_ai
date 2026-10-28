const test = require('node:test');
const assert = require('node:assert/strict');
const router = require('../src/services/router.service');
const confidence = require('../src/services/confidence.service');
const { redact } = require('../src/utils/redact');
const adapters = require('../src/config/adapters.json');
const piiCases = require('./fixtures/pii-cases.json');

const CONF = { minMeanLogprob: -0.9, lowTokenLogprob: -2.5, maxLowTokenRatio: 0.15 };

test('router: short simple question shrinks to tiny tier', () => {
  const d = router.route('What is the refund window?', adapters);
  assert.equal(d.tier, 'tiny');
  assert.deepEqual(
    d.adapters.map((a) => a.name),
    ['support'],
  );
  assert.equal('forceEscalate' in d, false);
});

test('router: reasoning-heavy question expands to core with code adapter', () => {
  const d = router.route('Explain why this node api query is slow and design an optimized architecture, step by step', adapters);
  assert.equal(d.tier, 'core');
  assert.equal(d.adapters[0].name, 'code');
});

test('router: short but hard tasks go to core', () => {
  assert.equal(router.route('Prove that this algorithm terminates', adapters).tier, 'core');
});

test('confidence: confident vs uncertain token streams', () => {
  const sure = Array.from({ length: 20 }, () => ({ logprob: -0.1 }));
  const unsure = Array.from({ length: 20 }, (_, i) => ({ logprob: i % 3 === 0 ? -4 : -0.6 }));
  assert.equal(confidence.assess(sure, CONF).confident, true);
  assert.equal(confidence.assess(unsure, CONF).confident, false);
  assert.equal(confidence.assess([], CONF).reason, 'no_logprobs');
});

test('redact: strips email, phone, card, PAN', () => {
  const out = redact('mail a.b@x.com call +91 98765 43210 card 4111 1111 1111 1111 pan ABCDE1234F');
  assert.ok(!out.includes('@x.com'));
  assert.ok(!out.includes('98765'));
  assert.ok(!out.includes('4111'));
  assert.ok(!out.includes('ABCDE1234F'));
});

// Shared with model/tests: the Python port must produce identical output for every case
test('redact: shared PII fixture (parity with model/viron/redact.py)', () => {
  for (const { input, expected } of piiCases) assert.equal(redact(input), expected, input);
});
