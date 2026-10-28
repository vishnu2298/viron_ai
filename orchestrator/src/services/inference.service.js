/**
 * Local inference against llama.cpp `llama-server` (OpenAI-compatible).
 * LoRA adapters are preloaded on the server with `--lora <file>` + `--lora-init-without-apply`
 * and selected per request via the `lora` field, so skills attach without reloading weights.
 */
const { postJson } = require('../utils/http');

function createInferenceService({ tiers, timeoutMs, http = postJson }) {
  async function generate({ tier, messages, adapters = [] }) {
    const t = tiers[tier];
    if (!t) throw new Error(`Unknown tier: ${tier}`);
    const body = {
      messages,
      max_tokens: t.maxTokens,
      temperature: 0.3,
      logprobs: true,
      top_logprobs: 1,
    };
    if (adapters.length) body.lora = adapters.map((a) => ({ id: a.id, scale: a.scale ?? 1 }));

    const res = await http(`${t.url}/v1/chat/completions`, body, { timeoutMs, service: `llm-${tier}` });
    const choice = res.choices?.[0] || {};
    return {
      text: choice.message?.content ?? '',
      logprobs: choice.logprobs?.content ?? [],
      usage: res.usage || {},
    };
  }
  return { generate };
}
module.exports = { createInferenceService };
