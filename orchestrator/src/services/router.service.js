/**
 * Router: decides how much model a request needs.
 *  - tier: 'tiny' | 'core'   (shrink vs expand compute)
 *  - adapters: LoRA skills to hot-attach
 * Deterministic + cheap (<1ms). Replace with a trained classifier once traces exist.
 */
// prettier-ignore
const REASONING_HINTS = [
  'why', 'explain', 'design', 'architecture', 'compare', 'analyze', 'analyse', 'prove',
  'step by step', 'trade-off', 'tradeoff', 'optimize', 'debug', 'plan', 'strategy',
];
// Clearly hard tasks go straight to the largest local tier
const HARD_HINTS = ['prove that', 'formal proof', 'legal advice', 'diagnose', 'multi-step plan with'];

// Whole-word match (Unicode-aware; JS \b is ASCII-only) with an optional plural, so 'api' no longer matches
// 'capital' and 'plan' no longer matches 'explanation', while 'bugs' still matches 'bug'.
const wordCache = new Map();
function hasWord(text, word) {
  let re = wordCache.get(word);
  if (!re) {
    const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    re = new RegExp(`(?<![\\p{L}\\p{M}\\p{N}])${escaped}(?:e?s)?(?![\\p{L}\\p{M}\\p{N}])`, 'u');
    wordCache.set(word, re);
  }
  return re.test(text);
}

function scoreComplexity(text) {
  const t = text.toLowerCase();
  const words = t.split(/\s+/).filter(Boolean).length;
  let score = Math.min(words / 60, 1) * 0.4;
  score += REASONING_HINTS.filter((h) => hasWord(t, h)).length * 0.15;
  if (/```|\bfunction\b|=>|\bclass\b/.test(text)) score += 0.2;
  if ((t.match(/\?/g) || []).length > 1) score += 0.1;
  return Math.min(score, 1);
}

function matchAdapters(text, registry, max = 2) {
  const t = text.toLowerCase();
  return registry
    .map((a) => ({ ...a, hits: a.keywords.filter((k) => hasWord(t, k.toLowerCase())).length }))
    .filter((a) => a.hits > 0)
    .sort((a, b) => b.hits - a.hits)
    .slice(0, max)
    .map(({ id, name, scale }) => ({ id, name, scale }));
}

function route(message, registry) {
  const complexity = scoreComplexity(message);
  const lower = message.toLowerCase();
  const hard = HARD_HINTS.some((h) => hasWord(lower, h));
  return {
    complexity: Number(complexity.toFixed(2)),
    tier: complexity < 0.35 && !hard ? 'tiny' : 'core',
    adapters: matchAdapters(message, registry),
  };
}

module.exports = { route, scoreComplexity, matchAdapters, hasWord };
