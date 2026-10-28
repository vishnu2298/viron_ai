const test = require('node:test');
const assert = require('node:assert/strict');
const router = require('../src/services/router.service');
const confidence = require('../src/services/confidence.service');
const { createRetrievalService } = require('../src/services/retrieval.service');
const { createInferenceService } = require('../src/services/inference.service');
const { createElasticService, buildSystem, SYSTEM_PROMPT } = require('../src/services/elastic.service');

const CONF = { minMeanLogprob: -0.9, lowTokenLogprob: -2.5, maxLowTokenRatio: 0.15 };
const sure = Array.from({ length: 8 }, () => ({ logprob: -0.1 }));
const unsure = Array.from({ length: 8 }, () => ({ logprob: -3 }));

// Records calls and returns a canned response, standing in for utils/http.postJson
function fakeHttp(response) {
  const calls = [];
  const http = async (url, body, opts) => {
    calls.push({ url, body, opts });
    return typeof response === 'function' ? response(url, body) : response;
  };
  return { http, calls };
}

test('retrieval: no knowledge repo -> empty context', async () => {
  const svc = createRetrievalService({ knowledgeRepo: null, embedUrl: 'http://e', topK: 3, minScore: 0.5 });
  assert.deepEqual(await svc.retrieve('q'), []);
});

test('retrieval: embedder configured -> vector search filtered by minScore', async () => {
  const { http, calls } = fakeHttp({ data: [{ embedding: [0.1, 0.2] }] });
  let searched;
  const knowledgeRepo = {
    vectorSearch: async (vector, k) => {
      searched = { vector, k };
      return [{ id: 'a', score: 0.9 }, { id: 'b', score: 0.2 }, { id: 'c' }];
    },
  };
  const svc = createRetrievalService({ knowledgeRepo, embedUrl: 'http://e', topK: 3, minScore: 0.5, timeoutMs: 100, http });
  const docs = await svc.retrieve('refunds');
  assert.deepEqual(
    docs.map((d) => d.id),
    ['a'],
  );
  assert.deepEqual(searched, { vector: [0.1, 0.2], k: 3 });
  assert.equal(calls[0].url, 'http://e/v1/embeddings');
  assert.deepEqual(calls[0].body, { input: 'refunds' });
});

test('retrieval: no embedder -> unfiltered text search', async () => {
  const knowledgeRepo = { textSearch: async (q, k) => [{ id: q, k, score: 0.01 }] };
  const svc = createRetrievalService({ knowledgeRepo, embedUrl: '', topK: 2, minScore: 0.5 });
  assert.deepEqual(await svc.retrieve('x'), [{ id: 'x', k: 2, score: 0.01 }]);
});

test('inference: unknown tier rejects; adapter scale defaults to 1; empty choice is safe', async () => {
  const { http, calls } = fakeHttp({});
  const svc = createInferenceService({ tiers: { tiny: { url: 'http://t', maxTokens: 5 } }, timeoutMs: 100, http });
  await assert.rejects(svc.generate({ tier: 'huge', messages: [] }), /Unknown tier/);
  const out = await svc.generate({ tier: 'tiny', messages: [], adapters: [{ id: 3 }] });
  assert.deepEqual(out, { text: '', logprobs: [], usage: {} });
  assert.deepEqual(calls[0].body.lora, [{ id: 3, scale: 1 }]);
  assert.equal(calls[0].url, 'http://t/v1/chat/completions');
});

test('confidence: non-finite logprobs only -> no_logprobs; many uncertain tokens flagged', () => {
  assert.equal(confidence.assess([{ logprob: null }, {}], CONF).reason, 'no_logprobs');
  const mixed = [...Array.from({ length: 16 }, () => ({ logprob: -0.05 })), ...Array.from({ length: 3 }, () => ({ logprob: -3 }))];
  assert.equal(confidence.assess(mixed, CONF).reason, 'too_many_uncertain_tokens');
});

test('router: code and multiple questions raise complexity; adapters are capped', () => {
  assert.ok(router.scoreComplexity('```x``` what? why?') > router.scoreComplexity('what'));
  const registry = [
    { id: 1, name: 'a', keywords: ['x'] },
    { id: 2, name: 'b', keywords: ['x', 'y'] },
    { id: 3, name: 'c', keywords: ['x'] },
  ];
  assert.deepEqual(
    router.matchAdapters('x y', registry).map((a) => a.name),
    ['b', 'a'],
  );
});

// Elastic pipeline with fully injected fakes
function elasticWith({ local = {}, retrieve, traceSave, route } = {}) {
  const calls = { local: [], traces: [] };
  const elastic = createElasticService({
    router: { route: route || ((m, reg) => router.route(m, reg)) },
    confidence,
    retrieval: { retrieve: retrieve || (async () => []) },
    inference: {
      generate: async ({ tier, messages }) => {
        calls.local.push({ tier, messages });
        const r = local[tier];
        if (r instanceof Error) throw r;
        return { text: `${tier} answer`, logprobs: r || sure };
      },
    },
    traceRepo: { save: traceSave || (async (t) => calls.traces.push(t)) },
    config: { adapters: [], confidence: CONF },
  });
  return { elastic, calls };
}

test('elastic: retrieval failure degrades to no context', async () => {
  const { elastic } = elasticWith({
    retrieve: async () => {
      throw new Error('mongo down');
    },
  });
  const out = await elastic.chat({ message: 'hi' });
  assert.equal(out.meta.servedBy, 'tiny');
  assert.equal(out.meta.contexts, 0);
});

test('elastic: options.tier=core skips tiny', async () => {
  const { elastic, calls } = elasticWith();
  const out = await elastic.chat({ message: 'hi', options: { tier: 'core' } });
  assert.equal(out.meta.servedBy, 'core');
  assert.deepEqual(
    calls.local.map((c) => c.tier),
    ['core'],
  );
});

test('elastic: hard prompts go straight to core', async () => {
  const { elastic, calls } = elasticWith();
  const out = await elastic.chat({ message: 'Prove that P != NP' });
  assert.equal(out.meta.servedBy, 'core');
  assert.deepEqual(
    calls.local.map((c) => c.tier),
    ['core'],
  );
});

test('elastic: unsure on both tiers -> core answer, not escalated', async () => {
  const { elastic } = elasticWith({ local: { tiny: unsure, core: unsure } });
  const out = await elastic.chat({ message: 'hi' });
  assert.equal(out.meta.servedBy, 'core');
  assert.equal(out.meta.escalated, false);
  assert.equal(out.answer, 'core answer');
});

test('elastic: tiny unsure and core down -> tiny answer is returned', async () => {
  const { elastic } = elasticWith({ local: { tiny: unsure, core: new Error('down') } });
  const out = await elastic.chat({ message: 'hi' });
  assert.equal(out.meta.servedBy, 'tiny');
  assert.equal(out.meta.attempts.at(-1).error, 'down');
});

test('elastic: every tier down -> 503 NO_TIER_AVAILABLE', async () => {
  const { elastic } = elasticWith({ local: { tiny: new Error('down'), core: new Error('down') } });
  await assert.rejects(elastic.chat({ message: 'hi' }), (err) => err.statusCode === 503 && err.code === 'NO_TIER_AVAILABLE');
});

test('elastic: trace save failure does not fail the request', async () => {
  const { elastic } = elasticWith({
    traceSave: async () => {
      throw new Error('write failed');
    },
  });
  const out = await elastic.chat({ message: 'hi' });
  assert.equal(out.answer, 'tiny answer');
  await new Promise((r) => setImmediate(r));
});

test('elastic: context goes to the system prompt; history and user turn stay verbatim', async () => {
  const history = [
    { role: 'user', content: 'earlier q' },
    { role: 'assistant', content: 'earlier a' },
  ];
  const { elastic, calls } = elasticWith({
    retrieve: async () => [{ id: 'k1', text: 'Fact one.' }],
    local: { tiny: unsure, core: unsure },
  });
  await elastic.chat({ message: 'new q', history });
  for (const { messages: sent } of calls.local) {
    assert.deepEqual(sent.slice(1), [...history, { role: 'user', content: 'new q' }]);
    assert.equal(sent[0].content, `${SYSTEM_PROMPT}\n\nContext:\n[1] Fact one.`);
  }
  assert.equal(calls.local.length, 2);
});

test('buildSystem: no context -> bare system prompt', () => {
  assert.equal(buildSystem([]), SYSTEM_PROMPT);
});
