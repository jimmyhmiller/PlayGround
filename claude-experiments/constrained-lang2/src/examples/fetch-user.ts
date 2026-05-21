// CLI demo: drive the fetch-user machine through a few scenarios and print
// the timeline to stdout. See fetch-user-inspect.ts for the live-debugger
// version.

import { Runtime } from '../core/runtime.ts';

import { fetchUserHandlers, fetchUserMachine } from './fetch-user-machine.ts';

const runtime = new Runtime(fetchUserMachine, {
  handlers: fetchUserHandlers,
  onTransition: (entry) => {
    const tag = entry.invariantFailure
      ? ` !INVARIANT(${entry.invariantFailure})`
      : '';
    // eslint-disable-next-line no-console
    console.log(
      `#${entry.seq} ${entry.prev.state} --${entry.event.type}--> ${entry.next.state}${tag}`,
    );
  },
});

async function main(): Promise<void> {
  runtime.send({ type: 'FETCH', userId: 'u1' });
  await wait(150);
  runtime.send({ type: 'RESET' });

  runtime.send({ type: 'FETCH', userId: 'does-not-exist' });
  await wait(150);
  runtime.send({ type: 'RESET' });

  const ignored = runtime.send({ type: 'RESET' });
  // eslint-disable-next-line no-console
  console.log(`\nignored RESET in Idle -> ${ignored === null ? 'null' : 'handled'}`);

  // eslint-disable-next-line no-console
  console.log('\nFinal state:', runtime.getState());
  // eslint-disable-next-line no-console
  console.log('Final context:', runtime.getContext());
  // eslint-disable-next-line no-console
  console.log(`\nTimeline length: ${runtime.getTimeline().length}`);
}

function wait(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

main().catch((err: unknown) => {
  // eslint-disable-next-line no-console
  console.error(err);
  process.exit(1);
});
