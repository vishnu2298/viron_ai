// Minimal local test of the worker with mocked bindings
import assert from 'node:assert/strict';
import worker, { cacheKey } from './src/index.js';

const kv = new Map();
const env = {
  ORIGIN_URL: 'http://127.0.0.1:1',
  CACHE_TTL_SECONDS: '60',
  FALLBACK_MODEL: 'm',
  CACHE: { get: async (k) => (kv.has(k) ? JSON.parse(kv.get(k)) : null), put: async (k, v) => kv.set(k, v) },
  AI: { run: async () => ({ response: 'fallback answer' }) },
};
const ctx = { waitUntil: (p) => p };
const call = (b) => worker.fetch(new Request('http://x/api/v1/chat', { method: 'POST', body: JSON.stringify(b) }), env, ctx);

assert.equal(await cacheKey('Hello  World'), await cacheKey('hello world'));
let r = await call({ message: 'hi' });
assert.equal(r.headers.get('x-elastic-tier'), 'edge-fallback');
kv.set(await cacheKey('cached q'), JSON.stringify({ data: { answer: 'from cache' }, meta: {} }));
r = await call({ message: 'Cached Q' });
assert.equal((await r.json()).data.answer, 'from cache');
assert.equal((await call({})).status, 400);
console.log('edge tests passed');
