// HTTP host: every request is one event; the matching reply effect becomes
// the response. Multiple in-flight requests are fine — correlation ids keep
// them separated.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';

import type { EffectBase, EventBase } from '../core/types.ts';
import { ReplyTimeoutError, type ReplyHost } from './reply.ts';

export interface HttpResponse {
  readonly status: number;
  readonly headers?: Readonly<Record<string, string>>;
  readonly body: string;
}

export interface StartHttpServerOptions<E extends EventBase> {
  readonly port: number;
  readonly host?: string;
  // Build the event from the parsed request. Body is the raw request body
  // as a string; user may JSON.parse if they want. Throw to short-circuit
  // with a 400 — the thrown message becomes the response body.
  readonly buildEvent: (
    req: IncomingMessage,
    body: string,
    correlationId: string,
  ) => E | Promise<E>;
  readonly formatResponse: (replyBody: unknown) => HttpResponse;
  readonly timeoutMs?: number;
  // Maximum request body size in bytes. Bodies larger than this get a 413.
  readonly maxBodyBytes?: number;
}

export interface RunningHttpServer {
  readonly url: string;
  stop(): Promise<void>;
}

export async function startHttpServer<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
>(
  host: ReplyHost<S, C, E, F>,
  opts: StartHttpServerOptions<E>,
): Promise<RunningHttpServer> {
  const bindHost = opts.host ?? '127.0.0.1';
  const maxBody = opts.maxBodyBytes ?? 1_000_000;

  const server = createServer((req, res) => {
    handleRequest(req, res, host, opts, maxBody).catch((err: unknown) => {
      writeError(res, 500, (err as Error).message);
    });
  });

  await new Promise<void>((resolve) => server.listen(opts.port, bindHost, () => resolve()));

  const url = `http://${bindHost}:${opts.port}`;
  // eslint-disable-next-line no-console
  console.log(`[http] listening on ${url}`);

  return {
    url,
    stop: () =>
      new Promise<void>((resolve) => {
        server.close(() => resolve());
      }),
  };
}

async function handleRequest<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
>(
  req: IncomingMessage,
  res: ServerResponse,
  host: ReplyHost<S, C, E, F>,
  opts: StartHttpServerOptions<E>,
  maxBody: number,
): Promise<void> {
  // Buffer the body. Real servers would stream; for a state-machine host
  // we want the full message before we can build an event.
  const body = await readBody(req, maxBody);
  if (body === null) {
    writeError(res, 413, 'request body too large');
    return;
  }

  let response: HttpResponse;
  try {
    response = await host.request(
      async (correlationId) => opts.buildEvent(req, body, correlationId),
      opts.timeoutMs,
    ).then((replyBody) => opts.formatResponse(replyBody));
  } catch (err: unknown) {
    if (err instanceof ReplyTimeoutError) {
      writeError(res, 504, err.message);
      return;
    }
    // buildEvent threw -> bad request. Anything else -> 500.
    writeError(res, 400, (err as Error).message);
    return;
  }

  const headers: Record<string, string> = {
    'content-type': 'application/json; charset=utf-8',
    ...response.headers,
  };
  res.writeHead(response.status, headers);
  res.end(response.body);
}

function readBody(req: IncomingMessage, maxBytes: number): Promise<string | null> {
  return new Promise((resolve, reject) => {
    let size = 0;
    const chunks: Buffer[] = [];
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > maxBytes) {
        req.destroy();
        resolve(null);
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function writeError(res: ServerResponse, status: number, message: string): void {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8' });
  res.end(JSON.stringify({ error: message }));
}
