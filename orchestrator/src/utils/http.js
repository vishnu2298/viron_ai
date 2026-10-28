const { UpstreamError } = require('./errors');

async function postJson(url, body, { headers = {}, timeoutMs = 60000, service = 'upstream' } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...headers },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    const text = await res.text();
    if (!res.ok) throw new UpstreamError(service, `HTTP ${res.status} ${text.slice(0, 300)}`);
    return JSON.parse(text);
  } catch (err) {
    if (err.name === 'AbortError') throw new UpstreamError(service, `timeout after ${timeoutMs}ms`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
module.exports = { postJson };
