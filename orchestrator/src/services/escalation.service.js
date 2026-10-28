/**
 * Escalation tier: only reached when the local tiers are not confident.
 * Every escalated answer is stored as a teacher label for adapter distillation.
 */
const { postJson } = require('../utils/http');

function createEscalationService({ apiKey, model, maxTokens, timeoutMs, http = postJson }) {
  async function generate({ system, messages }) {
    if (!apiKey) throw new Error('Escalation requested but ANTHROPIC_API_KEY is not set');
    const res = await http(
      'https://api.anthropic.com/v1/messages',
      { model, max_tokens: maxTokens, system, messages },
      {
        timeoutMs,
        service: 'escalation',
        headers: { 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
      },
    );
    const text = (res.content || [])
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join('\n');
    return { text, usage: res.usage || {} };
  }
  return { generate };
}
module.exports = { createEscalationService };
