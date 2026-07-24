"use client";

// A `"use client"` island in a real app-router app. In the react-server graph
// diffpack rewrites this to `createClientModuleProxy` client references (no
// component code reaches the flight render); in the client and SSR graphs it is
// bundled as real code and registered under a runtime id, so the flight's client
// reference resolves back to THIS component through the `__webpack_*` seam. It
// proves hydration (local `useState` increments) and the server-action round-trip
// (the `increment` server reference passed by the Server Component is invoked over
// `/_action/`).
import { useState } from "react";

export function Counter({
  initial,
  increment,
}: {
  initial: number;
  increment: (n: number) => Promise<number>;
}) {
  const [count, setCount] = useState(initial);
  const [serverResult, setServerResult] = useState<string>("none");

  return (
    <div id="island">
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
