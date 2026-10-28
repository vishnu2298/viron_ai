// Minimal local test of the worker with mocked bindings
import assert from 'node:assert/strict';
import http from 'node:http';
import worker, { cacheKey, shouldCache, timingSafeEqual } from './src/index.js';

const kv = new Map();
const env = {
  ORIGIN_URL: 'http://127.0.0.1:1',
  CACHE_TTL_SECONDS: '60',
  CACHE: { get: async (k) => (kv.has(k) ? JSON.parse(kv.get(k)) : null), put: async (k, v) => kv.set(k, v) },
};
// Any use of a Workers AI binding is a bug now. Non-enumerable so `{ ...env }` copies in tests skip it.
Object.defineProperty(env, 'AI', {
  get() {
    throw new Error('edge must not call Workers AI');
  },
});
const ctx = { waitUntil: (p) => p };
const call = (b) => worker.fetch(new Request('http://x/api/v1/chat', { method: 'POST', body: JSON.stringify(b) }), env, ctx);

assert.equal(await cacheKey('Hello  World'), await cacheKey('hello world'));
// Origin down: 503 with the standard error shape; cache hits are still served
let r = await call({ message: 'hi' });
assert.equal(r.status, 503);
assert.deepEqual(await r.json(), { success: false, message: 'Origin unavailable', code: 'ORIGIN_UNAVAILABLE' });
kv.set(await cacheKey('cached q'), JSON.stringify({ data: { answer: 'from cache' }, meta: {} }));
r = await call({ message: 'Cached Q' });
assert.equal(r.status, 200);
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

// Cache policy: only confident local answers or escalation answers
const ok = (meta) => ({ success: true, data: { answer: 'a' }, meta });
const att = (tier, confident) => ({ tier, confident });
assert.equal(shouldCache(200, ok({ servedBy: 'tiny', attempts: [att('tiny', true)] })), true);
assert.equal(shouldCache(200, ok({ servedBy: 'core', attempts: [att('tiny', false), att('core', true)] })), true);
assert.equal(shouldCache(200, ok({ servedBy: 'escalation', escalated: true, attempts: [att('tiny', false)] })), true);
assert.equal(shouldCache(200, ok({ servedBy: 'tiny', attempts: [att('tiny', false)] })), false);
// escalation failed after unsure local tiers: origin still answers from core, but it is not confident
assert.equal(
  shouldCache(200, ok({ servedBy: 'core', attempts: [att('tiny', false), att('core', false), { tier: 'escalation', error: 'x' }] })),
  false,
);
assert.equal(shouldCache(200, ok({ servedBy: 'core', attempts: [{ tier: 'core', error: 'down' }] })), false);
assert.equal(shouldCache(200, ok({ servedBy: 'tiny' })), false);
assert.equal(shouldCache(200, ok({ servedBy: 'edge-fallback', attempts: [] })), false);
assert.equal(shouldCache(200, ok({ servedBy: 'edge-cache', attempts: [att('tiny', true)] })), false);
assert.equal(shouldCache(200, { success: false, meta: { servedBy: 'tiny', attempts: [att('tiny', true)] } }), false);
assert.equal(shouldCache(429, ok({ servedBy: 'tiny', attempts: [att('tiny', true)] })), false);

// Integration: fake origin decides confidence per message; check what lands in KV
const origin = http.createServer((req, res) => {
  let raw = '';
  req.on('data', (c) => (raw += c));
  req.on('end', () => {
    const { message } = JSON.parse(raw);
    const confident = message.startsWith('sure');
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(ok({ servedBy: 'tiny', escalated: false, attempts: [att('tiny', confident)] })));
  });
});
await new Promise((resolve) => origin.listen(0, resolve));
const live = { ...env, ORIGIN_URL: `http://127.0.0.1:${origin.address().port}` };
const liveCall = (message) =>
  worker.fetch(new Request('http://x/api/v1/chat', { method: 'POST', body: JSON.stringify({ message }) }), live, ctx);
kv.clear();
assert.equal((await liveCall('sure thing')).headers.get('x-elastic-tier'), 'tiny');
assert.ok(kv.has(await cacheKey('sure thing')));
await liveCall('maybe thing');
assert.ok(!kv.has(await cacheKey('maybe thing')));
origin.close();

// Origin down -> 503 and nothing cached
kv.clear();
assert.equal((await call({ message: 'down q' })).status, 503);
assert.equal(kv.size, 0);

// Origin answers 5xx (e.g. NO_TIER_AVAILABLE) -> edge returns 503, nothing cached
const failing = http.createServer((req, res) => {
  req.resume();
  req.on('end', () => {
    res.statusCode = 503;
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({ success: false, message: 'No tier could produce an answer', code: 'NO_TIER_AVAILABLE' }));
  });
});
await new Promise((resolve) => failing.listen(0, resolve));
const failEnv = { ...env, ORIGIN_URL: `http://127.0.0.1:${failing.address().port}` };
r = await worker.fetch(new Request('http://x/api/v1/chat', { method: 'POST', body: JSON.stringify({ message: 'q5' }) }), failEnv, ctx);
assert.equal(r.status, 503);
assert.equal((await r.json()).code, 'ORIGIN_UNAVAILABLE');
assert.equal(kv.size, 0);
failing.close();

console.log('edge tests passed');
