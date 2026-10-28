// Pure view-model helpers for the live verification UI (unit-tested in test/ui.test.js; no DOM access here).

export const TIERS = ['tiny', 'core'];

/** Average per-token probability implied by a mean logprob, as a 0-100 percentage. */
export function tokenConfidencePct(meanLogprob) {
  if (typeof meanLogprob !== 'number' || !Number.isFinite(meanLogprob)) return null;
  return Math.round(Math.exp(meanLogprob) * 1000) / 10;
}

export function fmtMs(ms) {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return '—';
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

const REASONS = {
  ok: 'passed the confidence gate',
  low_mean_logprob: 'average token probability too low',
  too_many_uncertain_tokens: 'too many low-probability tokens',
  no_logprobs: 'model returned no logprobs',
};
export const reasonText = (reason) => REASONS[reason] || reason || '';

/**
 * One row per tier of the ladder the request could have used, in order.
 * state: confident | unsure | error | skipped (not needed or not on this request's ladder)
 */
export function ladder(meta) {
  const attempts = Array.isArray(meta?.attempts) ? meta.attempts : [];
  const start = meta?.tierRequested === 'core' ? 1 : 0;
  return TIERS.map((tier, i) => {
    const a = attempts.find((x) => x.tier === tier);
    const base = { tier, served: meta?.servedBy === tier, onLadder: i >= start };
    if (!a) return { ...base, state: 'skipped' };
    if (a.error) return { ...base, state: 'error', error: a.error, ms: a.ms };
    return {
      ...base,
      state: a.confident ? 'confident' : 'unsure',
      reason: a.reason,
      meanLogprob: a.meanLogprob,
      lowTokenRatio: a.lowTokenRatio,
      tokenConfidencePct: tokenConfidencePct(a.meanLogprob),
      ms: a.ms,
      adapters: a.adapters || [],
    };
  });
}

/** Overall verdict for an answer: did the tier that served it pass the confidence gate? */
export function verdict(meta) {
  const served = ladder(meta).find((s) => s.served);
  if (!served) return { level: 'none', label: 'No local answer' };
  if (served.state === 'confident') return { level: 'confident', label: `Confident · ${served.tier}` };
  return { level: 'unsure', label: `Best effort · ${served.tier} was unsure` };
}

/** Header stats from GET /api/v1/stats `data`. */
export function summarizeStats(data) {
  if (!data) return null;
  const rows = Array.isArray(data.byTier) ? data.byTier : [];
  const by = Object.fromEntries(rows.map((r) => [r._id, r.count]));
  return {
    total: data.total || 0,
    windowHours: data.windowHours,
    tiny: by.tiny || 0,
    core: by.core || 0,
    tinyShare: data.total ? Math.round(((by.tiny || 0) / data.total) * 100) : null,
    confidentShare: typeof data.localResolutionRate === 'number' ? Math.round(data.localResolutionRate * 100) : null,
  };
}

/** Conversation turns sent as `history`: last 20 completed user/assistant turns, as the API validates. */
export function historyFrom(messages) {
  return messages
    .filter((m) => (m.role === 'user' || m.role === 'assistant') && !m.pending && !m.failed && m.content)
    .slice(-20)
    .map(({ role, content }) => ({ role, content: content.slice(0, 8000) }));
}

/** Friendly text for API failures. */
export function errorText(status, body) {
  if (status === 401) return 'Unauthorized — set the API key (ORCHESTRATOR_API_KEY) in settings.';
  if (status === 429) return 'Rate limited — wait a moment and retry.';
  if (status === 503 || body?.code === 'NO_TIER_AVAILABLE') return 'No model tier is reachable. Is a Viron server running?';
  if (status === 400) return `Rejected: ${(body?.details || [body?.message]).filter(Boolean).join('; ')}`;
  return body?.message || `Request failed (HTTP ${status || 'network error'})`;
}
