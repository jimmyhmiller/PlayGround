import { Counter, Label } from "./Counter";

// Reference the client-boundary exports so they stay reachable and get a runtime id.
globalThis.__diffpack_client_test = { Counter, Label };

// Force a code-split so the build uses the registry runtime (and therefore the RSC
// `__webpack_*` seam), not the single-chunk scope-hoisted output.
import("./lazy").then((m) => {
  globalThis.__diffpack_lazy = m.value;
});
