/**
 * External Bridge via WebSocket
 *
 * Allows external processes to connect and receive events.
 * Protocol is JSON-based with message types for emit, query, and subscribe.
 */

import { WebSocketServer, WebSocket } from 'ws';
import { matchesPattern } from '../utils';
import type { EventStore } from '../EventStore';
import type { EventFilter, DashboardEvent } from '../../../types/events';

interface ExternalBridgeOptions {
  port?: number;
}

interface ExternalBridgeHandle {
  close(): void;
  getClientCount(): number;
}

interface WebSocketMessage {
  type: 'emit' | 'query' | 'subscribe';
  eventType?: string;
  payload?: unknown;
  filter?: EventFilter;
  pattern?: string;
  requestId?: string;
}

interface ExtendedWebSocket extends WebSocket {
  subscriptionPattern?: string;
}

/**
 * Setup WebSocket server for external event consumers
 */
export function setupExternalBridge(
  eventStore: EventStore,
  options: ExternalBridgeOptions = {}
): ExternalBridgeHandle {
  const port = options.port ?? 9876;

  const wss = new WebSocketServer({ host: '127.0.0.1', port });
  const clients = new Set<ExtendedWebSocket>();

  console.log(`[events] External bridge listening on ws://127.0.0.1:${port}`);

  wss.on('connection', (ws: ExtendedWebSocket) => {
    clients.add(ws);
    ws.subscriptionPattern = '**'; // Default: receive all events

    console.log('[events] External client connected');

    ws.on('message', (data: Buffer) => {
      try {
        const msg = JSON.parse(data.toString()) as WebSocketMessage;
        handleMessage(ws, msg, eventStore);
      } catch (_err) {
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

    ws.on('error', (err: Error) => {
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
  const unsubscribe = eventStore.subscribe('**', (event: DashboardEvent) => {
    const message = JSON.stringify({ type: 'event', event });
    for (const ws of clients) {
      if (ws.readyState === WebSocket.OPEN) {
        const pattern = ws.subscriptionPattern ?? '**';
        if (matchesPattern(event.type, pattern)) {
          ws.send(message);
        }
      }
    }
  });

  return {
    close(): void {
      unsubscribe();
      for (const ws of clients) {
        ws.close();
      }
      wss.close();
    },

    getClientCount(): number {
      return clients.size;
    },
  };
}

/**
 * Handle incoming WebSocket message
 */
function handleMessage(
  ws: ExtendedWebSocket,
  msg: WebSocketMessage,
  eventStore: EventStore
): void {
  switch (msg.type) {
    case 'emit': {
      // Emit event from external source
      if (!msg.eventType) {
        ws.send(JSON.stringify({ type: 'error', error: 'Missing eventType' }));
        return;
      }
      const event = eventStore.emit(msg.eventType, msg.payload ?? {}, {
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
    }

    case 'query': {
      // Query events
      const events = eventStore.getEvents(msg.filter ?? {});
      ws.send(
        JSON.stringify({
          type: 'query-result',
          requestId: msg.requestId,
          events,
        })
      );
      break;
    }

    case 'subscribe': {
      // Update subscription pattern
      ws.subscriptionPattern = msg.pattern ?? '**';
      ws.send(
        JSON.stringify({
          type: 'subscribed',
          pattern: ws.subscriptionPattern,
        })
      );
      break;
    }

    default:
      ws.send(
        JSON.stringify({
          type: 'error',
          error: `Unknown message type: ${msg.type}`,
        })
      );
  }
}
