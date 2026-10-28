const fs = require('fs');
const path = require('path');
const Joi = require('joi');

const num = (v, d) => (v === undefined || v === '' ? d : Number(v));

// Express `trust proxy`: 'true'/'false', a hop count, or a comma list of subnets/names (e.g. 'loopback')
function trustProxy(v) {
  if (v === undefined || v === '' || v === 'false') return false;
  if (v === 'true') return true;
  return /^\d+$/.test(v) ? Number(v) : v;
}

function loadAdapters(env) {
  const file = env.ADAPTERS_FILE || path.join(__dirname, 'adapters.json');
  if (!fs.existsSync(file)) return [];
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (err) {
    throw new Error(`Invalid config: ADAPTERS_FILE ${file} is not valid JSON (${err.message})`);
  }
}

function buildConfig(env) {
  return {
    port: num(env.PORT, 8080),
    env: env.NODE_ENV || 'development',
    apiKey: env.ORCHESTRATOR_API_KEY || '',
    trustProxy: trustProxy(env.TRUST_PROXY),
    // Only enable when the origin is reachable solely through Cloudflare; otherwise the header is spoofable
    trustCfIp: env.TRUST_CF_IP === 'true',

    // Elastic tiers: same model family at two sizes, each served by llama.cpp (OpenAI-compatible)
    tiers: {
      tiny: { url: env.TINY_MODEL_URL || 'http://localhost:8081', maxTokens: 256 },
      core: { url: env.CORE_MODEL_URL || 'http://localhost:8082', maxTokens: 1024 },
    },
    embedUrl: env.EMBED_URL || '',

    adapters: loadAdapters(env),

    confidence: {
      minMeanLogprob: num(env.CONF_MIN_MEAN_LOGPROB, -0.9),
      lowTokenLogprob: num(env.CONF_LOW_TOKEN_LOGPROB, -2.5),
      maxLowTokenRatio: num(env.CONF_MAX_LOW_TOKEN_RATIO, 0.15),
    },

    retrieval: { topK: num(env.RETRIEVAL_TOP_K, 4), minScore: num(env.RETRIEVAL_MIN_SCORE, 0.55) },

    mongo: { uri: env.MONGODB_URI || '', db: env.MONGODB_DB || 'viron_ai' },
    requestTimeoutMs: num(env.REQUEST_TIMEOUT_MS, 60000),
  };
}

const GENERATE_KEY =
  '"ORCHESTRATOR_API_KEY" must be a random secret of at least 32 characters when NODE_ENV=production ' +
  `(generate one: node -e "console.log(require('crypto').randomBytes(32).toString('hex'))")`;

// Labels are the env var names so a failed boot says exactly what to fix
const httpUrl = (label) =>
  Joi.string()
    .uri({ scheme: ['http', 'https'] })
    .label(label);
const tier = (label) => Joi.object({ url: httpUrl(label).required(), maxTokens: Joi.number().integer().min(1).required() });
const logprob = (label) => Joi.number().max(0).label(label).required();

const schema = Joi.object({
  port: Joi.number().port().label('PORT').required(),
  env: Joi.string().label('NODE_ENV').required(),
  // Production must never run open (empty key = dev mode) or with the .env.example placeholder
  apiKey: Joi.when('env', {
    is: 'production',
    then: Joi.string().min(32).invalid('change-me').required().messages({
      'any.required': GENERATE_KEY,
      'string.empty': GENERATE_KEY,
      'string.min': GENERATE_KEY,
      'any.invalid': GENERATE_KEY,
    }),
    otherwise: Joi.string().allow(''),
  }).label('ORCHESTRATOR_API_KEY'),
  trustProxy: Joi.alternatives(Joi.boolean(), Joi.number().integer().min(0), Joi.string().min(1)).label('TRUST_PROXY'),
  trustCfIp: Joi.boolean().label('TRUST_CF_IP'),
  tiers: Joi.object({ tiny: tier('TINY_MODEL_URL').required(), core: tier('CORE_MODEL_URL').required() }).required(),
  embedUrl: httpUrl('EMBED_URL').allow(''),
  adapters: Joi.array()
    .items(
      Joi.object({
        id: Joi.number().integer().min(0).required(),
        name: Joi.string().min(1).required(),
        scale: Joi.number().min(0).max(2),
        keywords: Joi.array().items(Joi.string().min(1)).min(1).required(),
      }).unknown(true),
    )
    .unique('id')
    .unique('name')
    .label('ADAPTERS_FILE'),
  confidence: Joi.object({
    minMeanLogprob: logprob('CONF_MIN_MEAN_LOGPROB'),
    lowTokenLogprob: logprob('CONF_LOW_TOKEN_LOGPROB'),
    maxLowTokenRatio: Joi.number().min(0).max(1).label('CONF_MAX_LOW_TOKEN_RATIO').required(),
  }).required(),
  retrieval: Joi.object({
    topK: Joi.number().integer().min(1).max(50).label('RETRIEVAL_TOP_K').required(),
    minScore: Joi.number().min(0).max(1).label('RETRIEVAL_MIN_SCORE').required(),
  }).required(),
  mongo: Joi.object({
    uri: Joi.string()
      .allow('')
      .pattern(/^mongodb(\+srv)?:\/\//)
      .label('MONGODB_URI'),
    db: Joi.string().min(1).label('MONGODB_DB').required(),
  }).required(),
  requestTimeoutMs: Joi.number().integer().min(1000).max(600_000).label('REQUEST_TIMEOUT_MS').required(),
});

/** Build config from env and fail fast, listing every invalid value at once. */
function loadConfig(env = process.env) {
  const { error, value } = schema.validate(buildConfig(env), { abortEarly: false, convert: false });
  if (error) throw new Error(`Invalid config: ${error.details.map((d) => d.message).join('; ')}`);
  return value;
}

module.exports = { loadConfig };
