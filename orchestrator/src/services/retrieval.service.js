/**
 * Knowledge lives outside the weights. Embeds the query (llama.cpp --embedding server)
 * and runs vector search; falls back to keyword search if no embedder is configured.
 */
const { postJson } = require('../utils/http');

function createRetrievalService({ knowledgeRepo, embedUrl, topK, minScore, timeoutMs, http = postJson }) {
  async function embed(text) {
    const res = await http(`${embedUrl}/v1/embeddings`, { input: text }, { timeoutMs, service: 'embedder' });
    return res.data?.[0]?.embedding;
  }

  async function retrieve(query) {
    if (!knowledgeRepo) return [];
    let docs;
    if (embedUrl) {
      const vector = await embed(query);
      docs = await knowledgeRepo.vectorSearch(vector, topK);
    } else {
      docs = await knowledgeRepo.textSearch(query, topK);
    }
    return embedUrl ? docs.filter((d) => (d.score ?? 0) >= minScore) : docs;
  }

  return { retrieve, embed };
}
module.exports = { createRetrievalService };
