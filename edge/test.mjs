// Minimal local test of the worker with mocked bindings
import assert from 'node:assert/strict';
import worker, { cacheKey, timingSafeEqual } from './src/index.js';

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

// Auth: constant-time compare of the full Authorization header
assert.equal(await timingSafeEqual('Bearer k1', 'Bearer k1'), true);
assert.equal(await timingSafeEqual('Bearer k1', 'Bearer k2'), false);
assert.equal(await timingSafeEqual('', 'Bearer k1'), false);
const authed = { ...env, EDGE_API_KEY: 'secret-key' };
const authCall = (authorization) => {
  const headers = authorization === undefined ? {} : { authorization };
  return worker.fetch(new Request('http://x/api/v1/chat', { method: 'POST', headers, body: '{}' }), authed, ctx).then((res) => res.status);
};
assert.equal(await authCall(), 401);
assert.equal(await authCall('Bearer secret-kez'), 401); // same length, last char differs
assert.equal(await authCall('Bearer secret-key-longer'), 401);
assert.equal(await authCall('secret-key'), 401);
assert.equal(await authCall('Bearer secret-key'), 400); // authorized; then fails validation on the empty body

console.log('edge tests passed');
