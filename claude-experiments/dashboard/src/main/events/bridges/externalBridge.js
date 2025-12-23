/**
 * External Bridge via WebSocket
 *
 * Allows external processes to connect and receive events.
 * Protocol is JSON-based with message types for emit, query, and subscribe.
 */

const { matchesPattern } = require('../utils');

/**
 * Setup WebSocket server for external event consumers
 *
 * @param {EventStore} eventStore
 * @param {Object} options
 * @param {number} options.port - Port to listen on (default: 9876)
 * @returns {Object} Bridge handle with close() method
 */
function setupExternalBridge(eventStore, options = {}) {
  const { WebSocketServer } = require('ws');
  const port = options.port || 9876;

  const wss = new WebSocketServer({ host: '127.0.0.1', port });
  const clients = new Set();

  console.log(`[events] External bridge listening on ws://127.0.0.1:${port}`);

  wss.on('connection', (ws) => {
    clients.add(ws);
    ws.subscriptionPattern = '**'; // Default: receive all events

    console.log('[events] External client connected');

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        handleMessage(ws, msg, eventStore);
      } catch (err) {
        ws.send(
          JSON.stringify({
            type: 'error',
            error: 'Invalid JSON message',
          })
        );
      }
    });

    ws.on('close', () => {
      clients.delete(ws);
      console.log('[events] External client disconnected');
    });

    ws.on('error', (err) => {
      console.error('[events] WebSocket error:', err);
      clients.delete(ws);
    });

    // Send welcome message
    ws.send(
      JSON.stringify({
        type: 'connected',
        sessionId: eventStore.sessionId,
      })
    );
  });

  // Push events to external clients
  const unsubscribe = eventStore.subscribe('**', (event) => {
    const message = JSON.stringify({ type: 'event', event });
    for (const ws of clients) {
      if (ws.readyState === 1) {
        // WebSocket.OPEN
        const pattern = ws.subscriptionPattern || '**';
        if (matchesPattern(event.type, pattern)) {
          ws.send(message);
        }
      }
    }
  });

  return {
    close() {
      unsubscribe();
      for (const ws of clients) {
        ws.close();
      }
      wss.close();
    },

    getClientCount() {
      return clients.size;
    },
  };
}

/**
 * Handle incoming WebSocket message
 * @private
 */
function handleMessage(ws, msg, eventStore) {
  switch (msg.type) {
    case 'emit':
      // Emit event from external source
      if (!msg.eventType) {
        ws.send(JSON.stringify({ type: 'error', error: 'Missing eventType' }));
        return;
      }
      const event = eventStore.emit(msg.eventType, msg.payload || {}, {
        source: 'external',
      });
      ws.send(
        JSON.stringify({
          type: 'emit-result',
          requestId: msg.requestId,
          event,
        })
      );
      break;

    case 'query':
      // Query events
      const events = eventStore.getEvents(msg.filter || {});
      ws.send(
        JSON.stringify({
          type: 'query-result',
          requestId: msg.requestId,
          events,
        })
      );
      break;

    case 'subscribe':
      // Update subscription pattern
      ws.subscriptionPattern = msg.pattern || '**';
      ws.send(
        JSON.stringify({
          type: 'subscribed',
          pattern: ws.subscriptionPattern,
        })
      );
      break;

    default:
      ws.send(
        JSON.stringify({
          type: 'error',
          error: `Unknown message type: ${msg.type}`,
        })
      );
  }
}

module.exports = { setupExternalBridge };
