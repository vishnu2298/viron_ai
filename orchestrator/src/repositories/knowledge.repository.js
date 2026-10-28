/**
 * MongoDB knowledge store.
 * Atlas: create a Vector Search index named `knowledge_vector` on field `embedding`.
 * Self-hosted / no embedder: a text index on `text` is used instead.
 */
function createKnowledgeRepository(db) {
  const col = db.collection('knowledge');

  return {
    async ensureIndexes() {
      await col.createIndex({ text: 'text', title: 'text' });
      await col.createIndex({ source: 1, updatedAt: -1 });
    },

    async upsert({ id, title, text, source, embedding, tags = [] }) {
      await col.updateOne(
        { _id: id },
        { $set: { title, text, source, embedding, tags, updatedAt: new Date() }, $setOnInsert: { createdAt: new Date() } },
        { upsert: true },
      );
    },

    async vectorSearch(vector, k) {
      return col
        .aggregate([
          { $vectorSearch: { index: 'knowledge_vector', path: 'embedding', queryVector: vector, numCandidates: k * 20, limit: k } },
          { $project: { _id: 1, title: 1, text: 1, source: 1, score: { $meta: 'vectorSearchScore' } } },
        ])
        .toArray()
        .then((docs) => docs.map((d) => ({ id: String(d._id), ...d })));
    },

    async textSearch(query, k) {
      return col
        .find({ $text: { $search: query } }, { projection: { title: 1, text: 1, source: 1, score: { $meta: 'textScore' } } })
        .sort({ score: { $meta: 'textScore' } })
        .limit(k)
        .toArray()
        .then((docs) => docs.map((d) => ({ id: String(d._id), ...d })));
    },
  };
}
module.exports = { createKnowledgeRepository };
