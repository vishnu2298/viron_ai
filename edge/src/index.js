/**
 * Edge tier (Cloudflare Worker):
 *  1. Auth at the edge.
 *  2. Exact-match answer cache in KV -> zero model compute for repeated questions.
 *  3. Forward to the orchestrator (tiny -> core -> escalation ladder).
 *  4. If the origin is down, degrade to a Workers AI small model.
 * Only stateless questions (no history) are cached, and only confident local or teacher answers.
 */
const json = (body, status = 200, extra = {}) =>
  new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...extra } });

export async function cacheKey(message) {
  const norm = message.toLowerCase().replace(/\s+/g, ' ').trim();
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(norm));
  return 'q:' + [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

function authorized(req, env) {
  if (!env.EDGE_API_KEY) return true;
  return (req.headers.get('authorization') || '') === `Bearer ${env.EDGE_API_KEY}`;
}

export default {
  async fetch(req, env, ctx) {
    const url = new URL(req.url);
    if (url.pathname === '/health') return json({ status: 'ok' });
    if (url.pathname !== '/api/v1/chat' || req.method !== 'POST') return json({ success: false, code: 'NOT_FOUND' }, 404);
    if (!authorized(req, env)) return json({ success: false, code: 'UNAUTHORIZED' }, 401);

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
      if (key && out.success) {
        ctx.waitUntil(
          env.CACHE.put(key, JSON.stringify({ data: out.data, meta: out.meta }), {
            expirationTtl: Number(env.CACHE_TTL_SECONDS || 3600),
          }),
        );
      }
      return json(out, res.status, { 'x-elastic-tier': out.meta?.servedBy || 'origin' });
    } catch (err) {
      const ai = await env.AI.run(env.FALLBACK_MODEL, {
        messages: [...(body.history || []), { role: 'user', content: body.message }],
        max_tokens: 512,
      });
      return json(
        {
          success: true,
          data: { answer: ai.response },
          meta: { servedBy: 'edge-fallback', degraded: true, reason: String(err.message || err) },
        },
        200,
        { 'x-elastic-tier': 'edge-fallback' },
      );
    }
  },
};
