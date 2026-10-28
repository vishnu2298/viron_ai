const { MongoClient } = require('mongodb');
const logger = require('./utils/logger');
const router = require('./services/router.service');
const confidence = require('./services/confidence.service');
const { createInferenceService } = require('./services/inference.service');
const { createEscalationService } = require('./services/escalation.service');
const { createRetrievalService } = require('./services/retrieval.service');
const { createElasticService } = require('./services/elastic.service');
const { createKnowledgeRepository } = require('./repositories/knowledge.repository');
const { createTraceRepository } = require('./repositories/trace.repository');
const { createApp } = require('./app');
const { createShutdown, createPendingTracker } = require('./shutdown');

async function main() {
  const config = require('./config'); // validates on load; a bad value aborts startup via the catch below
  let mongo, knowledgeRepo, traceRepo;
  const pending = createPendingTracker();
  if (config.mongo.uri) {
    mongo = await new MongoClient(config.mongo.uri, { maxPoolSize: 20 }).connect();
    const db = mongo.db(config.mongo.db);
    knowledgeRepo = createKnowledgeRepository(db);
    const traces = createTraceRepository(db);
    traceRepo = { ...traces, save: (trace) => pending.track(traces.save(trace)) };
    await Promise.all([knowledgeRepo.ensureIndexes(), traces.ensureIndexes()]);
  } else {
    logger.warn({ event: 'mongo.disabled', note: 'no MONGODB_URI: retrieval and tracing off' });
  }

  const elastic = createElasticService({
    router,
    confidence,
    retrieval: createRetrievalService({ knowledgeRepo, embedUrl: config.embedUrl, ...config.retrieval, timeoutMs: 10_000 }),
    inference: createInferenceService({ tiers: config.tiers, timeoutMs: config.requestTimeoutMs }),
    escalation: createEscalationService({ ...config.escalation, timeoutMs: config.requestTimeoutMs }),
    traceRepo,
    config,
  });

  const server = createApp({ elastic, traceRepo, config }).listen(config.port, () =>
    logger.info({ event: 'server.started', port: config.port, adapters: config.adapters.map((a) => a.name) }),
  );

  const shutdown = createShutdown({ server, mongo, pending });
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

main().catch((err) => {
  logger.error({ event: 'server.fatal', error: err.message });
  process.exit(1);
});
