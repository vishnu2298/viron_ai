/**
 * Edge tier (Cloudflare Worker):
 *  1. Auth at the edge.
 *  2. Exact-match answer cache in KV -> zero model compute for repeated questions.
 *  3. Forward to the orchestrator (tiny -> core ladder).
 *  4. If the origin is down, return 503; cached answers are still served. No external model is ever called.
 * Only stateless questions (no history) are cached, and only answers the local confidence gate passed.
 */
const json = (body, status = 200, extra = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...extra } });

export async function cacheKey(message) {
  const norm = message.toLowerCase().replace(/\s+/g, ' ').trim();
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(norm));
  return 'q:' + [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

const sha256 = async (s) => new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s)));

// Compare fixed-length digests with no early exit, so timing reveals neither the key's length nor its prefix.
export async function timingSafeEqual(a, b) {
  const [x, y] = await Promise.all([sha256(a), sha256(b)]);
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

const LOCAL_TIERS = new Set(['tiny', 'core']);

// Cache only answers worth replaying: a tiny/core answer whose attempt passed the confidence gate.
// Unsure answers (the ladder's best-effort fallback) and anything not from a local tier are never cached.
export function shouldCache(status, out) {
  if (status < 200 || status >= 300 || out?.success !== true) return false;
  const meta = out.meta || {};
  if (!LOCAL_TIERS.has(meta.servedBy) || !Array.isArray(meta.attempts)) return false;
  const served = meta.attempts.findLast((a) => a.tier === meta.servedBy && !a.error);
  return served?.confident === true;
}

async function authorized(req, env) {
  if (!env.EDGE_API_KEY) return true;
  return timingSafeEqual(req.headers.get('authorization') || '', `Bearer ${env.EDGE_API_KEY}`);
}

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    if (url.pathname === '/health') return json({ status: 'ok' });
    if (url.pathname !== '/api/v1/chat' || req.method !== 'POST') return json({ success: false, code: 'NOT_FOUND' }, 404);
    if (!(await authorized(req, env))) return json({ success: false, code: 'UNAUTHORIZED' }, 401);

    let body;
    try {
      body = await req.json();
    } catch {
      return json({ success: false, code: 'BAD_JSON' }, 400);
    }
    if (typeof body?.message !== 'string' || !body.message.trim() || body.message.length > 8000) {
      return json({ success: false, code: 'VALIDATION_ERROR' }, 400);
    }

    const cacheable = !body.history?.length;
    const key = cacheable ? await cacheKey(body.message) : null;
    if (key) {
      const hit = await env.CACHE.get(key, 'json');
      if (hit)
        return json({ success: true, data: hit.data, meta: { ...hit.meta, servedBy: 'edge-cache' } }, 200, { 'x-elastic-tier': 'cache' });
    }

    try {
      const res = await fetch(`${env.ORIGIN_URL}/api/v1/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${env.ORIGIN_API_KEY || ''}` },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(45000),
      });
      if (res.status >= 500) throw new Error(`origin ${res.status}`);
      const out = await res.json();
      if (key && shouldCache(res.status, out)) {
        ctx.waitUntil(
          env.CACHE.put(key, JSON.stringify({ data: out.data, meta: out.meta }), {
            expirationTtl: Number(env.CACHE_TTL_SECONDS || 3600),
          }),
        );
      }
      return json(out, res.status, { 'x-elastic-tier': out.meta?.servedBy || 'origin' });
    } catch (err) {
      // Standalone: no substitute model at the edge. Internal detail stays in logs, not in the response.
      console.error(JSON.stringify({ event: 'origin.unavailable', error: String(err.message || err) }));
      return json({ success: false, message: 'Origin unavailable', code: 'ORIGIN_UNAVAILABLE' }, 503, { 'retry-after': '5' });
    }
  },
};
