// Navigation-completeness demo layout. A `"use client"` layout that reads
// `useSelectedLayoutSegment()` to highlight the active child section, exercising the
// SEGMENT_BOUNDARY island the adapter wraps around each layout. It also renders
// `<Link prefetch>` targets: hover/focus warms the client Router's prefetch cache so a
// subsequent click swaps instantly.
"use client";

import Link from "next/link";
import { useSelectedLayoutSegment, useSelectedLayoutSegments } from "next/navigation";

export default function NavDemoLayout({ children }: { children: React.ReactNode }) {
  const segment = useSelectedLayoutSegment();
  const segments = useSelectedLayoutSegments();
  return (
    <section id="nav-demo">
      <nav id="nav-demo-nav">
        <Link
          id="link-alpha"
          href="/nav-demo/alpha"
          prefetch
          className={segment === "alpha" ? "active" : ""}
        >
          alpha
        </Link>
        <Link
          id="link-beta"
          href="/nav-demo/beta"
          className={segment === "beta" ? "active" : ""}
        >
          beta
        </Link>
      </nav>
      <p id="active-segment">segment: {segment === null ? "(none)" : segment}</p>
      <p id="active-segments">segments: {segments.join(",")}</p>
      <div id="nav-demo-children">{children}</div>
    </section>
  );
}
