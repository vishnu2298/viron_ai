/**
 * The Viron AI pipeline — shrink by default, expand only when needed:
 *   route → retrieve → tiny/core (+LoRA) → confidence gate → expand to core → escalate
 * Each request is traced; escalations become distillation data for new adapters.
 */
const logger = require('../utils/logger');
const { redact } = require('../utils/redact');

const SYSTEM_PROMPT =
  'You are a concise, accurate assistant. Use the provided context when relevant. ' +
  'If the context does not contain the answer and you are unsure, say so plainly.';

// Retrieved context rides in the (single, leading) system message so user turns and history stay clean.
// One system message, not two: several chat templates reject a system message that is not first/only.
function buildSystem(contexts) {
  if (!contexts.length) return SYSTEM_PROMPT;
  return `${SYSTEM_PROMPT}\n\nContext:\n${contexts.map((c, i) => `[${i + 1}] ${c.text}`).join('\n')}`;
}

function buildMessages(message, history) {
  return [...history, { role: 'user', content: message }];
}

function createElasticService({ router, retrieval, inference, escalation, confidence, traceRepo, config }) {
  async function tryLocal(tier, system, messages, adapters, attempts) {
    const started = Date.now();
    const out = await inference.generate({
      tier,
      adapters,
      messages: [{ role: 'system', content: system }, ...messages],
    });
    const conf = confidence.assess(out.logprobs, config.confidence);
    attempts.push({ tier, adapters: adapters.map((a) => a.name), ms: Date.now() - started, ...conf });
    return { ...out, conf };
  }

  async function chat({ message, history = [], options = {} }) {
    const t0 = Date.now();
    const decision = router.route(message, config.adapters);
    if (options.tier) decision.tier = options.tier;

    let contexts = [];
    try {
      contexts = await retrieval.retrieve(message);
    } catch (err) {
      logger.warn({ event: 'retrieval.failed', error: err.message }); // degrade, don't fail
    }
    const system = buildSystem(contexts);
    const messages = buildMessages(message, history);
    const attempts = [];
    let answer = null;
    let servedBy = null;

    if (!decision.forceEscalate) {
      const ladder = decision.tier === 'tiny' ? ['tiny', 'core'] : ['core'];
      for (const tier of ladder) {
        try {
          const out = await tryLocal(tier, system, messages, decision.adapters, attempts);
          answer = out.text;
          servedBy = tier;
          if (out.conf.confident) break;
        } catch (err) {
          attempts.push({ tier, error: err.message });
          logger.warn({ event: 'local.failed', tier, error: err.message });
        }
      }
    }

    const lastConfident = attempts.length && attempts[attempts.length - 1].confident;
    const needEscalation = decision.forceEscalate || !lastConfident;
    let escalated = false;

    if (needEscalation && config.escalation.enabled && options.allowEscalation !== false) {
      try {
        const out = await escalation.generate({ system, messages });
        answer = out.text;
        servedBy = 'escalation';
        escalated = true;
      } catch (err) {
        attempts.push({ tier: 'escalation', error: err.message });
        logger.error({ event: 'escalation.failed', error: err.message });
      }
    }

    if (answer === null) {
      const e = new Error('No tier could produce an answer');
      e.statusCode = 503;
      e.code = 'NO_TIER_AVAILABLE';
      throw e;
    }

    const trace = {
      message: redact(message),
      answer: redact(answer),
      servedBy,
      escalated,
      decision,
      attempts,
      contextIds: contexts.map((c) => c.id),
      latencyMs: Date.now() - t0,
      createdAt: new Date(),
    };
    if (traceRepo) traceRepo.save(trace).catch((err) => logger.warn({ event: 'trace.save_failed', error: err.message }));

    return {
      answer,
      meta: {
        servedBy,
        escalated,
        tierRequested: decision.tier,
        complexity: decision.complexity,
        adapters: decision.adapters.map((a) => a.name),
        contexts: contexts.length,
        attempts,
        latencyMs: trace.latencyMs,
      },
    };
  }

  return { chat };
}

module.exports = { createElasticService, buildMessages, buildSystem, SYSTEM_PROMPT };
