"use client";

// A `"use client"` island in a real app-router app. In the react-server graph
// diffpack rewrites this to `createClientModuleProxy` client references (no
// component code reaches the flight render); in the client and SSR graphs it is
// bundled as real code and registered under a runtime id, so the flight's client
// reference resolves back to THIS component through the `__webpack_*` seam. It
// proves hydration (local `useState` increments) and the server-action round-trip
// (the `increment` server reference passed by the Server Component is invoked over
// `/_action/`).
// `useId` is here as a HYDRATION SEAM PROBE, not because the island needs an id.
// React derives it from the tree-id fork a multi-child parent pushes — NOT from the
// rendered markup — so if the SSR entry and the client entry wrap the flight root in
// even slightly different shapes, every useId under the tree silently disagrees across
// the seam while the HTML still looks identical. Rendering it into an attribute is what
// makes that comparable from outside: SSR value vs post-hydration value.
import { useId, useState } from "react";

export function Counter({
  initial,
  increment,
}: {
  initial: number;
  increment: (n: number) => Promise<number>;
}) {
  const [count, setCount] = useState(initial);
  const [serverResult, setServerResult] = useState<string>("none");
  const seamId = useId();

  return (
    <div id="island" data-uid={seamId}>
      <span id="counter">count: {count}</span>
      <button id="inc" onClick={() => setCount((current) => current + 1)}>
        inc
      </button>
      <button
        id="server-inc"
        onClick={async () => {
          const result = await increment(count);
          setServerResult(String(result));
        }}
      >
        server-inc
      </button>
      <span id="server-result">server: {serverResult}</span>
    </div>
  );
}
