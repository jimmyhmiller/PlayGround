import { defineWorld } from "../../../../src/world/index.js";

/**
 * The SvelteKit demo app is stateless server-side: the Sverdle game state
 * lives in a cookie, and bat gives every flow a fresh browser context, so
 * the world is empty by construction. reset() is a real no-op, not a stub.
 */
export default defineWorld({
  reset: () => {
    // nothing to do: no server-side world; per-flow contexts isolate cookies
  },
  entities: {},
});
