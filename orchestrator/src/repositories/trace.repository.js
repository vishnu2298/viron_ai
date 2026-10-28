/**
 * Request traces = the system's learning signal.
 * Redacted rows feed training/export_traces.py (LoRA datasets for Viron checkpoints, roadmap M5).
 */
function createTraceRepository(db) {
  const col = db.collection('traces');
  return {
    async ensureIndexes() {
      await col.createIndex({ confident: 1, createdAt: -1 });
      await col.createIndex({ 'decision.adapters.name': 1, createdAt: -1 });
      await col.createIndex({ createdAt: 1 }, { expireAfterSeconds: 60 * 60 * 24 * 90 }); // 90-day retention
    },
    async save(trace) {
      await col.insertOne(trace);
    },
    async stats(sinceHours = 24) {
      const since = new Date(Date.now() - sinceHours * 3600 * 1000);
      return col
        .aggregate([
          { $match: { createdAt: { $gte: since } } },
          {
            $group: {
              _id: '$servedBy',
              count: { $sum: 1 },
              // traces written before the `confident` field existed count as not confident
              confident: { $sum: { $cond: [{ $eq: ['$confident', true] }, 1, 0] } },
              avgLatencyMs: { $avg: '$latencyMs' },
            },
          },
          { $sort: { count: -1 } },
        ])
        .toArray();
    },
  };
}
module.exports = { createTraceRepository };
