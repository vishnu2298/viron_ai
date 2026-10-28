function createChatController({ elastic, traceRepo }) {
  return {
    async chat(req, res, next) {
      try {
        const { answer, meta } = await elastic.chat(req.body);
        res.json({ success: true, message: 'ok', data: { answer }, meta });
      } catch (err) {
        next(err);
      }
    },
    async stats(req, res, next) {
      try {
        const hours = Math.min(Number(req.query.hours) || 24, 24 * 30);
        const rows = traceRepo ? await traceRepo.stats(hours) : [];
        const total = rows.reduce((a, r) => a + r.count, 0);
        // Standalone runtime: "resolved locally" = answered by a tier that passed the confidence gate
        const confident = rows.reduce((a, r) => a + (r.confident || 0), 0);
        res.json({
          success: true,
          message: 'ok',
          data: { windowHours: hours, total, localResolutionRate: total ? Number((confident / total).toFixed(3)) : null, byTier: rows },
        });
      } catch (err) {
        next(err);
      }
    },
  };
}
module.exports = { createChatController };
