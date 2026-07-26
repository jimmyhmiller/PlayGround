// A `"use client"` island proving router.refresh() is a SOFT RSC refresh, not a full
// reload: the local counter state MUST survive a refresh (which re-fetches the server
// component's flight and diff-renders it, keeping the document mounted). The refreshed
// server timestamp arrives as `children` (the server component re-renders on refresh).
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function RefreshControls({ children }: { children: React.ReactNode }) {
  const [count, setCount] = useState(0);
  const router = useRouter();
  return (
    <div id="refresh-controls">
      <span id="refresh-count">count: {count}</span>
      <button id="bump" onClick={() => setCount((c) => c + 1)}>
        bump
      </button>
      <button id="do-refresh" onClick={() => router.refresh()}>
        refresh
      </button>
      <div id="server-value">{children}</div>
    </div>
  );
}
