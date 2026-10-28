/**
 * Confidence gate from token logprobs (OpenAI-compatible `logprobs.content[]`).
 * Low mean logprob or many low-probability tokens => the small model is guessing.
 */
function assess(logprobContent, cfg) {
  if (!Array.isArray(logprobContent) || logprobContent.length === 0) {
    return { confident: false, reason: 'no_logprobs', meanLogprob: null, lowTokenRatio: null };
  }
  const lps = logprobContent.map((t) => t.logprob).filter((v) => Number.isFinite(v));
  if (lps.length === 0) return { confident: false, reason: 'no_logprobs', meanLogprob: null, lowTokenRatio: null };

  const mean = lps.reduce((a, b) => a + b, 0) / lps.length;
  const lowRatio = lps.filter((v) => v < cfg.lowTokenLogprob).length / lps.length;

  let reason = 'ok';
  if (mean < cfg.minMeanLogprob) reason = 'low_mean_logprob';
  else if (lowRatio > cfg.maxLowTokenRatio) reason = 'too_many_uncertain_tokens';

  return {
    confident: reason === 'ok',
    reason,
    meanLogprob: Number(mean.toFixed(3)),
    lowTokenRatio: Number(lowRatio.toFixed(3)),
  };
}
module.exports = { assess };
