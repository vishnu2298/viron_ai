const crypto = require('crypto');
module.exports = (apiKey) => (req, res, next) => {
  if (!apiKey) return next(); // dev mode
  const given = (req.headers.authorization || '').replace(/^Bearer\s+/i, '');
  const a = Buffer.from(given);
  const b = Buffer.from(apiKey);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) {
    return res.status(401).json({ success: false, message: 'Unauthorized', code: 'UNAUTHORIZED' });
  }
  next();
};
