const logger = require('../utils/logger');
module.exports = (env) => (err, req, res, next) => { // eslint-disable-line no-unused-vars
  const status = err.statusCode || 500;
  logger.error({ event: 'request.error', url: req.originalUrl, code: err.code, error: err.message });
  res.status(status).json({
    success: false,
    message: status >= 500 && env === 'production' ? 'Internal server error' : err.message,
    code: err.code || 'INTERNAL_ERROR',
  });
};
