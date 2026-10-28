const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { createShutdown, createPendingTracker } = require('../src/shutdown');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Server whose handler takes `delayMs`, so a request is in flight when shutdown starts
function slowServer(delayMs) {
  const server = http.createServer(async (req, res) => {
    await sleep(delayMs);
    res.end('done');
  });
  return new Promise((r) => server.listen(0, () => r({ server, url: `http://127.0.0.1:${server.address().port}` })));
}

function fakes() {
  const events = [];
  return {
    events,
    mongo: { close: async () => events.push('mongo.close') },
    exit: (code) => events.push(`exit:${code}`),
  };
}

test('shutdown: in-flight request completes, then pending writes flush, then Mongo closes, exit 0', async () => {
  const { server, url } = await slowServer(80);
  const { events, mongo, exit } = fakes();
  const pending = createPendingTracker();
  pending.track(sleep(250).then(() => events.push('trace.saved')));

  const inFlight = fetch(url).then(async (r) => {
    events.push(`response:${r.status}`);
    return r.text();
  });
  await sleep(20); // request is now being handled
  const done = createShutdown({ server, mongo, pending, timeoutMs: 2000, exit })('SIGTERM');

  assert.equal(await inFlight, 'done');
  await done;
  assert.deepEqual(events, ['response:200', 'trace.saved', 'mongo.close', 'exit:0']);
  await assert.rejects(fetch(url)); // no longer accepting connections
});

test('shutdown: drain timeout force-closes sockets and exits 1', async () => {
  const { server, url } = await slowServer(5000);
  const { events, mongo, exit } = fakes();
  const hung = fetch(url).catch(() => 'socket closed');
  await sleep(20);
  await createShutdown({ server, mongo, timeoutMs: 50, exit })('SIGTERM');
  assert.deepEqual(events, ['exit:1']);
  assert.equal(await hung, 'socket closed');
});

test('shutdown: second signal is ignored; works without Mongo', async () => {
  const { server } = await slowServer(0);
  const { events, exit } = fakes();
  const shutdown = createShutdown({ server, timeoutMs: 1000, exit });
  await Promise.all([shutdown('SIGTERM'), shutdown('SIGINT')]);
  assert.deepEqual(events, ['exit:0']);
});

test('shutdown: Mongo close failure exits 1', async () => {
  const { server } = await slowServer(0);
  const events = [];
  const mongo = {
    close: async () => {
      throw new Error('boom');
    },
  };
  await createShutdown({ server, mongo, timeoutMs: 1000, exit: (c) => events.push(c) })('SIGTERM');
  assert.deepEqual(events, [1]);
});

test('pending tracker: settles rejected writes and empties', async () => {
  const pending = createPendingTracker();
  pending.track(Promise.reject(new Error('write failed'))).catch(() => {});
  pending.track(sleep(10));
  assert.equal(pending.size, 2);
  await pending.drain();
  await sleep(0);
  assert.equal(pending.size, 0);
});
