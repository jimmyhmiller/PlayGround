// HTTP deployment: the SAME agent machine exposed as POST /ask.
//   curl -X POST localhost:5174/ask -d '{"question":"hello"}'
// Same machine, same handlers, different host.

import { ReplyHost } from '../host/reply.ts';
import { startHttpServer } from '../host/http.ts';
import { agentHandlers, agentMachine } from './agent-machine.ts';

const host = new ReplyHost(agentMachine, agentHandlers);

await startHttpServer(host, {
  port: 5174,
  buildEvent: (req, body, correlationId) => {
    if (req.method !== 'POST') {
      throw new Error(`only POST is supported, got ${req.method}`);
    }
    let payload: { question?: unknown };
    try {
      payload = JSON.parse(body || '{}') as { question?: unknown };
    } catch {
      throw new Error('request body must be JSON');
    }
    if (typeof payload.question !== 'string') {
      throw new Error('payload.question must be a string');
    }
    return {
      type: 'Ask' as const,
      question: payload.question,
      correlationId,
    };
  },
  formatResponse: (replyBody) => {
    const b = replyBody as { answer?: string; error?: string };
    if (b.error !== undefined) {
      return {
        status: 500,
        body: JSON.stringify({ error: b.error }),
      };
    }
    return {
      status: 200,
      body: JSON.stringify({ answer: b.answer ?? null }),
    };
  },
  timeoutMs: 5000,
});

// eslint-disable-next-line no-console
console.log(`try:  curl -s -X POST localhost:5174/ask -d '{"question":"hello"}'`);
