const net = require('net');
const express = require('express');
const rateLimit = require('express-rate-limit');
const { validate, chatSchema } = require('../middleware/validate');

// Behind Cloudflare every request arrives from a Cloudflare edge IP, so key on the real client instead.
// req.ip already honours `trust proxy`; the CF header is used only when explicitly trusted and well-formed.
const rateLimitKey =
  ({ trustCfIp = false } = {}) =>
  (req) => {
    const cf = req.headers['cf-connecting-ip'];
    return trustCfIp && typeof cf === 'string' && net.isIP(cf) ? cf : req.ip;
  };

module.exports = (controller, { trustCfIp = false } = {}) => {
  const r = express.Router();
  const limiter = rateLimit({
    windowMs: 60_000,
    limit: 60,
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: rateLimitKey({ trustCfIp }),
  });
  r.post('/chat', limiter, validate(chatSchema), controller.chat);
  r.get('/stats', controller.stats);
  return r;
};
module.exports.rateLimitKey = rateLimitKey;
