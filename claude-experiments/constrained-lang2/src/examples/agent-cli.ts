// CLI deployment: the agent in a REPL.
// Run with: npm run agent-cli

import { ReplyHost } from '../host/reply.ts';
import { runCli } from '../host/cli.ts';
import { agentHandlers, agentMachine } from './agent-machine.ts';

const host = new ReplyHost(agentMachine, agentHandlers);

// eslint-disable-next-line no-console
console.log(`agent ready — type a question and press enter. Ctrl-C to exit.`);

await runCli(host, {
  prompt: 'you> ',
  buildEvent: (line, correlationId) => ({
    type: 'Ask' as const,
    question: line,
    correlationId,
  }),
  formatReply: (body) => {
    const b = body as { answer?: string; error?: string };
    if (b.error !== undefined) return `agent> [ERROR] ${b.error}`;
    return `agent> ${b.answer ?? '(no answer)'}`;
  },
});
