const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { loadConfig } = require('../src/config/load');

test('config: defaults are valid', () => {
  const c = loadConfig({});
  assert.equal(c.port, 8080);
  assert.equal(c.trustProxy, false);
  assert.equal(c.escalation.enabled, true);
  assert.ok(c.adapters.length > 0);
});

test('config: TRUST_PROXY / TRUST_CF_IP parsing', () => {
  assert.equal(loadConfig({ TRUST_PROXY: '' }).trustProxy, false);
  assert.equal(loadConfig({ TRUST_PROXY: 'true' }).trustProxy, true);
  assert.equal(loadConfig({ TRUST_PROXY: '1' }).trustProxy, 1);
  assert.equal(loadConfig({ TRUST_PROXY: 'loopback, 10.0.0.0/8' }).trustProxy, 'loopback, 10.0.0.0/8');
  assert.equal(loadConfig({ TRUST_CF_IP: 'true' }).trustCfIp, true);
  assert.equal(loadConfig({ TRUST_CF_IP: 'yes' }).trustCfIp, false);
});

test('config: every bad value is reported at once, by env var name', () => {
  assert.throws(
    () =>
      loadConfig({
        PORT: '70000',
        TINY_MODEL_URL: 'ftp://x',
        EMBED_URL: 'not a url',
        CONF_MIN_MEAN_LOGPROB: '0.5',
        CONF_MAX_LOW_TOKEN_RATIO: 'abc',
        RETRIEVAL_TOP_K: '0',
        RETRIEVAL_MIN_SCORE: '2',
        ESCALATION_MAX_TOKENS: '1.5',
        MONGODB_URI: 'postgres://x',
        REQUEST_TIMEOUT_MS: '10',
      }),
    (err) => {
      for (const name of [
        'PORT',
        'TINY_MODEL_URL',
        'EMBED_URL',
        'CONF_MIN_MEAN_LOGPROB',
        'CONF_MAX_LOW_TOKEN_RATIO',
        'RETRIEVAL_TOP_K',
        'RETRIEVAL_MIN_SCORE',
        'ESCALATION_MAX_TOKENS',
        'MONGODB_URI',
        'REQUEST_TIMEOUT_MS',
      ]) {
        assert.match(err.message, new RegExp(`"${name}"`), name);
      }
      return true;
    },
  );
});

test('config: accepts valid overrides', () => {
  const c = loadConfig({
    PORT: '3000',
    EMBED_URL: 'http://embedder:8083',
    MONGODB_URI: 'mongodb+srv://cluster.example.net',
    TRUST_PROXY: '2',
    RETRIEVAL_MIN_SCORE: '0',
  });
  assert.equal(c.port, 3000);
  assert.equal(c.retrieval.minScore, 0);
});

function withAdapters(content, fn) {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'viron-cfg-')), 'adapters.json');
  fs.writeFileSync(file, content);
  try {
    return fn(file);
  } finally {
    fs.rmSync(path.dirname(file), { recursive: true });
  }
}

test('config: adapter registry is validated', () => {
  const dupId = JSON.stringify([
    { id: 0, name: 'a', keywords: ['x'] },
    { id: 0, name: 'b', keywords: ['y'] },
  ]);
  withAdapters(dupId, (f) => assert.throws(() => loadConfig({ ADAPTERS_FILE: f }), /ADAPTERS_FILE.*duplicate/));
  withAdapters('[{"id":0,"name":"a","keywords":[]}]', (f) => assert.throws(() => loadConfig({ ADAPTERS_FILE: f }), /keywords/));
  withAdapters('{oops', (f) => assert.throws(() => loadConfig({ ADAPTERS_FILE: f }), /not valid JSON/));
  assert.deepEqual(loadConfig({ ADAPTERS_FILE: '/nonexistent/adapters.json' }).adapters, []);
});
