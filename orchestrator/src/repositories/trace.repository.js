/**
 * Request traces = the system's learning signal.
 * Escalated rows are teacher-labelled examples; training/export_traces.py turns them into LoRA datasets.
 */
function createTraceRepository(db) {
  const col = db.collection('traces');
  return {
    async ensureIndexes() {
      await col.createIndex({ escalated: 1, createdAt: -1 });
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
          { $group: { _id: '$servedBy', count: { $sum: 1 }, avgLatencyMs: { $avg: '$latencyMs' } } },
          { $sort: { count: -1 } },
        ])
        .toArray();
    },
  };
}
module.exports = { createTraceRepository };
