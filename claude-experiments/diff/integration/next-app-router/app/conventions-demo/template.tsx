"use client";
// The `template.tsx` convention: a layout-like wrapper that RE-MOUNTS on navigation
// (fresh state per URL), unlike `layout.tsx` which preserves state. diffpack composes
// it just inside this segment's layout, keyed by pathname so React remounts it whenever
// the URL changes. A `useState` initialized from a module-load timestamp proves the
// remount (a fresh mount reruns the initializer). The wrapper div carries a stable
// marker the SSR smoke test asserts on.
import { useState } from "react";

export default function Template({ children }: { children: React.ReactNode }) {
  const [mountId] = useState(() => Math.random().toString(36).slice(2));
  return (
    <div id="conventions-template" data-mount={mountId}>
      <p>template-wrapper</p>
      {children}
    </div>
  );
}
