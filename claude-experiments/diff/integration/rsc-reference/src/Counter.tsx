"use client";

// The interactive `"use client"` island. In the REACT-SERVER graph diffpack
// rewrites this module to `createClientModuleProxy` client references (no component
// code reaches the flight render); in the CLIENT and SSR graphs it is bundled as
// REAL code and registered in diffpack's registry under a runtime id, so both the
// SSR-of-flight pass and the browser resolve the flight's client reference back to
// THIS component through the `__webpack_*` seam.
//
// It proves two things the Slice E browser oracle asserts:
//   • hydration + interactivity: `useState` local count increments on click;
//   • the server-action round-trip: the `increment` server reference the Server
//     Component passed as a prop is invoked over `/_action/` and its result shown.
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
