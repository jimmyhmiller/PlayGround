// CLI host: read lines from stdin, push each as an event into the machine,
// print the reply. One in-flight request at a time (sequential REPL), which
// matches how a CLI agent loop actually behaves.

import readline from 'node:readline';
import { stdin, stdout } from 'node:process';

import type { EffectBase, EventBase } from '../core/types.ts';
import type { ReplyHost } from './reply.ts';

export interface RunCliOptions<E extends EventBase> {
  readonly prompt?: string;
  readonly buildEvent: (line: string, correlationId: string) => E;
  readonly formatReply: (body: unknown) => string;
  readonly timeoutMs?: number;
  // Called when a request rejects (timeout, machine error, etc.). Default
  // prints the error and continues the loop.
  readonly onError?: (err: Error) => void;
}

export async function runCli<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
>(
  host: ReplyHost<S, C, E, F>,
  opts: RunCliOptions<E>,
): Promise<void> {
  const isTty = stdin.isTTY === true;
  // terminal: true breaks piped/non-tty input — it tries to do raw-mode
  // control sequences. Auto-detect.
  const rl = readline.createInterface({
    input: stdin,
    output: stdout,
    terminal: isTty,
  });
  const prompt = opts.prompt ?? '> ';
  const onError =
    opts.onError ??
    ((err) => {
      // eslint-disable-next-line no-console
      console.error(`[error] ${err.message}`);
    });

  const cleanup = (): void => {
    host.shutdown();
    rl.close();
  };
  process.once('SIGINT', () => {
    // eslint-disable-next-line no-console
    console.log('\n[bye]');
    cleanup();
    process.exit(0);
  });

  if (isTty) rl.setPrompt(prompt);
  if (isTty) rl.prompt();

  // The async iterator from readline buffers all pending lines and only
  // ends after every line has been delivered, so piped input never gets
  // dropped between iterations. We await each request before moving on
  // to the next line — strict REPL semantics.
  for await (const raw of rl) {
    const line = raw.trim();
    if (line === '') {
      if (isTty) rl.prompt();
      continue;
    }
    try {
      const body = await host.request(
        (correlationId) => opts.buildEvent(line, correlationId),
        opts.timeoutMs,
      );
      // eslint-disable-next-line no-console
      console.log(opts.formatReply(body));
    } catch (err: unknown) {
      onError(err as Error);
    }
    if (isTty) rl.prompt();
  }
  cleanup();
}
