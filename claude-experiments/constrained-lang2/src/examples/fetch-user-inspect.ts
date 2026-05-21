// Inspector demo: launch the fetch-user machine attached to the live
// debugger UI. The machine starts in Idle and waits for events sent from
// the browser at http://127.0.0.1:5173 (or via console.log helpers below).

import { Runtime } from '../core/runtime.ts';
import { Inspector } from '../inspector/server.ts';

import { fetchUserHandlers, fetchUserMachine } from './fetch-user-machine.ts';

const runtime = new Runtime(fetchUserMachine, {
  handlers: fetchUserHandlers,
});

const inspector = new Inspector(runtime, { port: 5173 });

await inspector.start();

// eslint-disable-next-line no-console
console.log(`
  Machine is idle. Open the inspector and click event chips to drive it.
  Try: FETCH (then edit JSON to add userId), then HTTP_OK or HTTP_ERR
  arrives automatically as a reply event.

  Example FETCH:  { "type": "FETCH", "userId": "u1" }
  Example RESET:  { "type": "RESET" }

  Press Ctrl-C to stop.
`);
