/**
 * Graceful shutdown: stop accepting connections, let in-flight requests finish, flush fire-and-forget
 * trace writes, then close Mongo. A hard timeout (below Docker's 10s SIGKILL grace) bounds the drain.
 */
const logger = require('./utils/logger');

// Tracks fire-and-forget promises (trace writes) so shutdown can wait for them before closing Mongo
function createPendingTracker() {
  const inFlight = new Set();
  return {
    track(promise) {
      inFlight.add(promise);
      promise.then(
        () => inFlight.delete(promise),
        () => inFlight.delete(promise),
      );
      return promise;
    },
    drain: () => Promise.allSettled([...inFlight]),
    get size() {
      return inFlight.size;
    },
  };
}

function createShutdown({ server, mongo, pending, timeoutMs = 8000, exit = process.exit }) {
  let started = false;

  async function drain() {
    // Keep-alive sockets that go idle after their in-flight response would otherwise hold close() open
    const sweep = setInterval(() => server.closeIdleConnections(), 100);
    try {
      await new Promise((resolve) => {
        server.close(resolve);
        server.closeIdleConnections();
      });
    } finally {
      clearInterval(sweep);
    }
    if (pending) await pending.drain();
    if (mongo) await mongo.close();
  }

  return async function shutdown(signal) {
    if (started) return;
    started = true;
    logger.info({ event: 'server.stopping', signal, pendingWrites: pending?.size ?? 0 });

    let timer;
    const timeout = new Promise((resolve) => {
      timer = setTimeout(() => resolve('timeout'), timeoutMs);
    });
    try {
      const result = await Promise.race([drain().then(() => 'drained'), timeout]);
      if (result === 'timeout') {
        logger.warn({ event: 'server.drain_timeout', timeoutMs });
        server.closeAllConnections();
        return exit(1);
      }
      logger.info({ event: 'server.stopped' });
      return exit(0);
    } catch (err) {
      logger.error({ event: 'server.shutdown_failed', error: err.message });
      return exit(1);
    } finally {
      clearTimeout(timer);
    }
  };
}

module.exports = { createShutdown, createPendingTracker };
