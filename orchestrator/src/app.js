const express = require('express');
const helmet = require('helmet');
const auth = require('./middleware/auth');
const errorHandler = require('./middleware/error');
const v1 = require('./routes/v1');
const { createChatController } = require('./controllers/chat.controller');

function createApp({ elastic, traceRepo, config }) {
  const app = express();
  app.disable('x-powered-by');
  app.set('trust proxy', config.trustProxy ?? false);
  app.use(helmet());
  app.use(express.json({ limit: '256kb' }));
  app.get('/health', (req, res) => res.json({ status: 'ok' }));
  app.use('/api/v1', auth(config.apiKey), v1(createChatController({ elastic, traceRepo }), { trustCfIp: config.trustCfIp }));
  app.use((req, res) => res.status(404).json({ success: false, message: 'Not found', code: 'NOT_FOUND' }));
  app.use(errorHandler(config.env));
  return app;
}
module.exports = { createApp };
