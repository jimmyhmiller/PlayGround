"use client";

import { useState } from "react";

// A CLIENT component used as JSX inside an MDX route: the MDX page is a server
// component, so this also exercises the client boundary being crossed from MDX.
export default function Counter() {
  const [count, setCount] = useState(0);
  return (
    <button type="button" data-testid="counter" onClick={() => setCount(count + 1)}>
      clicked {count} times
    </button>
  );
}
