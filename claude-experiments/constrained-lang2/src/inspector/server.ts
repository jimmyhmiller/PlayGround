// Inspector: HTTP + WebSocket bridge between a Runtime and the live debugger UI.
//
// Usage:
//   const inspector = new Inspector(runtime, { port: 5173 });
//   await inspector.start();
//   // -> open http://localhost:5173
//
// Responsibilities:
//   - serve the static UI at /
//   - accept WS connections at /ws
//   - on connect: send 'init' with the full machine snapshot + timeline so far
//   - on every transition: broadcast a 'transition' message
//   - on incoming 'send': dispatch the event into the runtime
//
// The Inspector does NOT own the machine; it just observes and pokes one.
// Multiple clients can connect at once.

import { createReadStream, statSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { WebSocketServer, type WebSocket } from 'ws';

import type { Runtime } from '../core/runtime.ts';
import { MachineFrozenError } from '../core/runtime.ts';
import type { EffectBase, EventBase, TimelineEntry } from '../core/types.ts';

import type { ClientMessage, MachineSnapshot, ServerMessage } from './protocol.ts';

const __dirname = dirname(fileURLToPath(import.meta.url));
const UI_DIR = resolve(__dirname, 'ui');

export interface InspectorOptions {
  readonly port?: number;
  readonly host?: string;
}

export class Inspector<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
> {
  private readonly clients = new Set<WebSocket>();
  private readonly port: number;
  private readonly host: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private httpServer?: any;
  private wss?: WebSocketServer;
  private unsubscribe?: () => void;

  constructor(
    private readonly runtime: Runtime<S, C, E, F>,
    opts: InspectorOptions = {},
  ) {
    this.port = opts.port ?? 5173;
    this.host = opts.host ?? '127.0.0.1';
  }

  async start(): Promise<{ url: string }> {
    const server = createServer((req, res) => this.serveStatic(req, res));
    const wss = new WebSocketServer({ server, path: '/ws' });

    wss.on('connection', (ws) => {
      this.clients.add(ws);
      ws.send(JSON.stringify(this.buildInit()));

      ws.on('message', (data) => {
        try {
          const msg = JSON.parse(data.toString()) as ClientMessage;
          this.handleClientMessage(ws, msg);
        } catch (err: unknown) {
          this.sendError(ws, `bad message: ${(err as Error).message}`);
        }
      });
      ws.on('close', () => {
        this.clients.delete(ws);
      });
    });

    this.unsubscribe = this.runtime.subscribe((entry) => {
      this.broadcast({
        type: 'transition',
        entry: entry as TimelineEntry<string, unknown, EventBase, EffectBase>,
        state: this.runtime.getState(),
        context: this.runtime.getContext(),
        frozen: this.runtime.isFrozen(),
      });
    });

    await new Promise<void>((resolveStart) => {
      server.listen(this.port, this.host, () => resolveStart());
    });

    this.httpServer = server;
    this.wss = wss;

    const url = `http://${this.host}:${this.port}`;
    // eslint-disable-next-line no-console
    console.log(`[inspector] listening on ${url}`);
    return { url };
  }

  async stop(): Promise<void> {
    this.unsubscribe?.();
    for (const ws of this.clients) ws.close();
    await new Promise<void>((res) => this.wss?.close(() => res()));
    await new Promise<void>((res) => this.httpServer?.close(() => res()));
  }

  // --- internals --------------------------------------------------------

  private buildInit(): ServerMessage {
    const def = this.runtime.def;
    const snapshot: MachineSnapshot = {
      name: def.name,
      initial: def.initial,
      states: (Object.keys(def.states) as S[]).map((name) => ({
        name,
        accepts: Object.keys(def.states[name].on),
      })),
      effectKinds: [...def.effectKinds],
      invariants: def.invariants.map((i) => i.name),
    };
    return {
      type: 'init',
      machine: snapshot,
      state: this.runtime.getState(),
      context: this.runtime.getContext(),
      timeline: this.runtime.getTimeline() as ReadonlyArray<
        TimelineEntry<string, unknown, EventBase, EffectBase>
      >,
      frozen: this.runtime.isFrozen(),
    };
  }

  private handleClientMessage(ws: WebSocket, msg: ClientMessage): void {
    switch (msg.type) {
      case 'send': {
        // Cast: the runtime knows its own E; the wire carries arbitrary
        // EventBase. The runtime will harmlessly ignore unknown event types
        // (no handler), or invariants will fire if the event corrupts state.
        try {
          this.runtime.send(msg.event as E);
        } catch (err: unknown) {
          if (err instanceof MachineFrozenError) {
            this.sendError(ws, 'machine is frozen (invariant violated)');
          } else {
            this.sendError(ws, (err as Error).message);
          }
        }
        return;
      }
    }
  }

  private broadcast(msg: ServerMessage): void {
    const json = JSON.stringify(msg);
    for (const ws of this.clients) {
      if (ws.readyState === ws.OPEN) ws.send(json);
    }
  }

  private sendError(ws: WebSocket, message: string): void {
    const msg: ServerMessage = { type: 'error', message };
    ws.send(JSON.stringify(msg));
  }

  // Minimal static file server for the UI. Only serves files under UI_DIR;
  // path traversal attempts get a 403.
  private serveStatic(req: IncomingMessage, res: ServerResponse): void {
    const url = req.url ?? '/';
    const relPath = url === '/' ? 'index.html' : url.replace(/^\/+/, '');
    const filePath = resolve(UI_DIR, relPath);
    if (!filePath.startsWith(UI_DIR + '/') && filePath !== join(UI_DIR, 'index.html')) {
      res.statusCode = 403;
      res.end('forbidden');
      return;
    }
    try {
      const stat = statSync(filePath);
      if (!stat.isFile()) {
        res.statusCode = 404;
        res.end('not found');
        return;
      }
      const contentType = filePath.endsWith('.html')
        ? 'text/html; charset=utf-8'
        : filePath.endsWith('.js')
          ? 'text/javascript; charset=utf-8'
          : filePath.endsWith('.css')
            ? 'text/css; charset=utf-8'
            : 'application/octet-stream';
      res.setHeader('content-type', contentType);
      createReadStream(filePath).pipe(res);
    } catch {
      res.statusCode = 404;
      res.end('not found');
    }
  }
}
