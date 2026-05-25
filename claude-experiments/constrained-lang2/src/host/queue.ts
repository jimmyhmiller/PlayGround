// Queue worker host: pull messages from any async iterable, push each as an
// event, forward the reply to a sink. Concurrency is bounded; messages within
// a batch run in parallel.
//
// The source is an AsyncIterable<Msg>, which is the lowest-common-denominator
// for queues: SQS, Redis BLPOP, Kafka consumer, an in-memory channel, etc.
// can all be wrapped as one.

import type { EffectBase, EventBase } from '../core/types.ts';
import type { ReplyHost } from './reply.ts';

export interface RunQueueWorkerOptions<Msg, E extends EventBase> {
  readonly source: AsyncIterable<Msg>;
  readonly buildEvent: (msg: Msg, correlationId: string) => E;
  readonly onReply: (msg: Msg, body: unknown) => void | Promise<void>;
  readonly onError?: (msg: Msg, err: Error) => void | Promise<void>;
  readonly timeoutMs?: number;
  // Max in-flight messages. Default 1 (strict FIFO). Increase for parallel
  // processing of independent jobs.
  readonly concurrency?: number;
}

export async function runQueueWorker<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
  Msg,
>(
  host: ReplyHost<S, C, E, F>,
  opts: RunQueueWorkerOptions<Msg, E>,
): Promise<void> {
  const concurrency = Math.max(1, opts.concurrency ?? 1);
  const onError =
    opts.onError ??
    ((msg, err) => {
      // eslint-disable-next-line no-console
      console.error(`[queue] message failed:`, msg, err.message);
    });

  // Simple semaphore: keep up to `concurrency` in-flight promises.
  const inFlight = new Set<Promise<void>>();

  for await (const msg of opts.source) {
    const job = processOne(host, msg, opts, onError).finally(() => {
      inFlight.delete(job);
    });
    inFlight.add(job);
    if (inFlight.size >= concurrency) {
      // Wait for at least one to finish before pulling the next message.
      await Promise.race(inFlight);
    }
  }
  // Drain remaining.
  await Promise.all(inFlight);
}

async function processOne<
  S extends string,
  C,
  E extends EventBase,
  F extends EffectBase,
  Msg,
>(
  host: ReplyHost<S, C, E, F>,
  msg: Msg,
  opts: RunQueueWorkerOptions<Msg, E>,
  onError: (msg: Msg, err: Error) => void | Promise<void>,
): Promise<void> {
  try {
    const body = await host.request(
      (correlationId) => opts.buildEvent(msg, correlationId),
      opts.timeoutMs,
    );
    await opts.onReply(msg, body);
  } catch (err: unknown) {
    await onError(msg, err as Error);
  }
}

// --- Small helper: a hand-fed in-memory queue, useful for tests and demos.

export class MemoryQueue<Msg> implements AsyncIterable<Msg> {
  private readonly buffer: Msg[] = [];
  private waiters: Array<(value: IteratorResult<Msg>) => void> = [];
  private closed = false;

  push(msg: Msg): void {
    if (this.closed) throw new Error('queue is closed');
    const waiter = this.waiters.shift();
    if (waiter !== undefined) {
      waiter({ value: msg, done: false });
    } else {
      this.buffer.push(msg);
    }
  }

  close(): void {
    this.closed = true;
    for (const w of this.waiters) w({ value: undefined as Msg, done: true });
    this.waiters = [];
  }

  [Symbol.asyncIterator](): AsyncIterator<Msg> {
    return {
      next: () => {
        const m = this.buffer.shift();
        if (m !== undefined) return Promise.resolve({ value: m, done: false });
        if (this.closed) return Promise.resolve({ value: undefined as Msg, done: true });
        return new Promise<IteratorResult<Msg>>((resolve) => {
          this.waiters.push(resolve);
        });
      },
    };
  }
}
