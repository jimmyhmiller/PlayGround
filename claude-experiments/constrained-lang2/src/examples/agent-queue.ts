// Queue-worker deployment: the SAME agent machine consuming an in-memory
// queue. In production the source would be SQS/Redis/Kafka/etc. — anything
// you can wrap as an async iterable plugs in unchanged.

import { ReplyHost } from '../host/reply.ts';
import { MemoryQueue, runQueueWorker } from '../host/queue.ts';
import { agentHandlers, agentMachine } from './agent-machine.ts';

interface Job {
  id: string;
  question: string;
}

const host = new ReplyHost(agentMachine, agentHandlers);
const queue = new MemoryQueue<Job>();

// Push three jobs, then close the queue so the worker drains and exits.
queue.push({ id: 'j1', question: 'hello' });
queue.push({ id: 'j2', question: 'this one will fail please' });
queue.push({ id: 'j3', question: 'goodbye' });
queue.close();

await runQueueWorker(host, {
  source: queue,
  concurrency: 2, // process two jobs in parallel; agent supports it
  buildEvent: (job, correlationId) => ({
    type: 'Ask' as const,
    question: job.question,
    correlationId,
  }),
  onReply: (job, body) => {
    const b = body as { answer?: string; error?: string };
    // eslint-disable-next-line no-console
    console.log(
      `  [job ${job.id}] reply: ${b.error !== undefined ? `ERROR ${b.error}` : b.answer}`,
    );
  },
});

// eslint-disable-next-line no-console
console.log('all jobs drained.');
