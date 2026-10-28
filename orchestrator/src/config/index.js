const fs = require('fs');
const path = require('path');

const num = (v, d) => (v === undefined || v === '' ? d : Number(v));

function loadAdapters() {
  const file = process.env.ADAPTERS_FILE || path.join(__dirname, 'adapters.json');
  return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : [];
}

module.exports = {
  port: num(process.env.PORT, 8080),
  env: process.env.NODE_ENV || 'development',
  apiKey: process.env.ORCHESTRATOR_API_KEY || '',

  // Elastic tiers: same model family at two sizes, each served by llama.cpp (OpenAI-compatible)
  tiers: {
    tiny: { url: process.env.TINY_MODEL_URL || 'http://localhost:8081', maxTokens: 256 },
    core: { url: process.env.CORE_MODEL_URL || 'http://localhost:8082', maxTokens: 1024 },
  },
  embedUrl: process.env.EMBED_URL || '',

  adapters: loadAdapters(),

  confidence: {
    minMeanLogprob: num(process.env.CONF_MIN_MEAN_LOGPROB, -0.9),
    lowTokenLogprob: num(process.env.CONF_LOW_TOKEN_LOGPROB, -2.5),
    maxLowTokenRatio: num(process.env.CONF_MAX_LOW_TOKEN_RATIO, 0.15),
  },

  retrieval: { topK: num(process.env.RETRIEVAL_TOP_K, 4), minScore: num(process.env.RETRIEVAL_MIN_SCORE, 0.55) },

  escalation: {
    enabled: process.env.ESCALATION_ENABLED !== 'false',
    apiKey: process.env.ANTHROPIC_API_KEY || '',
    model: process.env.ESCALATION_MODEL || 'claude-sonnet-5-5',
    maxTokens: num(process.env.ESCALATION_MAX_TOKENS, 1024),
  },

  mongo: { uri: process.env.MONGODB_URI || '', db: process.env.MONGODB_DB || 'viron_ai' },
  requestTimeoutMs: num(process.env.REQUEST_TIMEOUT_MS, 60000),
};
