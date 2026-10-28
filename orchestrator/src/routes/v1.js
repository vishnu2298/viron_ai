const express = require('express');
const rateLimit = require('express-rate-limit');
const { validate, chatSchema } = require('../middleware/validate');

module.exports = (controller) => {
  const r = express.Router();
  r.post(
    '/chat',
    rateLimit({ windowMs: 60_000, limit: 60, standardHeaders: true, legacyHeaders: false }),
    validate(chatSchema),
    controller.chat,
  );
  r.get('/stats', controller.stats);
  return r;
};
