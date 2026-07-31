// Next's pages-router ROUTE ANNOUNCER — the visually hidden live region every
// pages-router app ships (`next/dist/client/route-announcer` rendered through
// `next/dist/client/portal` from `next/dist/client/index`).
//
// After every client-side route change it announces the new page to a screen
// reader, choosing the announcement the way Next does: the document title, else
// the first `<h1>`, else the path. It is NOT rendered on the first load (a screen
// reader announces a document load by itself) and it is NOT server-rendered: the
// portal node is created in an effect, so the hydrated markup is unchanged.
//
// This is real, user-observable Next behaviour, not decoration — an app built by
// Next has `<next-route-announcer>` in its body after a navigation and a diffpack
// build did not, which is a difference an assistive technology user perceives.

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "./next-router.jsx";

// Next's `nextjsRouteAnnouncerStyles`: off-screen but still announced (a
// `display:none` region is skipped by screen readers).
const ANNOUNCER_STYLE = {
  border: 0,
  clip: "rect(0 0 0 0)",
  height: "1px",
  margin: "-1px",
  overflow: "hidden",
  padding: 0,
  position: "absolute",
  top: 0,
  width: "1px",
  whiteSpace: "nowrap",
  wordWrap: "normal",
};

export function RouteAnnouncer() {
  const { asPath } = useRouter();
  const [announcement, setAnnouncement] = useState("");
  // Seeded with the path this mounted at, so the FIRST load announces nothing.
  const announcedPath = useRef(asPath);

  useEffect(() => {
    if (announcedPath.current === asPath) return;
    announcedPath.current = asPath;
    // Next's priority: document title, then the first h1, then the path itself.
    if (document.title) {
      setAnnouncement(document.title);
    } else {
      const heading = document.querySelector("h1");
      const content = heading ? (heading.innerText ?? heading.textContent) : null;
      setAnnouncement(content || asPath);
    }
  }, [asPath]);

  return (
    <p
      aria-live="assertive"
      id="__next-route-announcer__"
      role="alert"
      style={ANNOUNCER_STYLE}
    >
      {announcement}
    </p>
  );
}

// The `<next-route-announcer>` host element, appended to `document.body` (outside
// the hydrated `#__next` container) exactly as Next's `Portal` does. Renders null
// until the effect has created the node, so server markup and the first client
// render agree and hydration is untouched.
export default function RouteAnnouncerPortal() {
  const [host, setHost] = useState(null);

  useEffect(() => {
    const element = document.createElement("next-route-announcer");
    document.body.appendChild(element);
    setHost(element);
    return () => {
      document.body.removeChild(element);
    };
  }, []);

  return host ? createPortal(<RouteAnnouncer />, host) : null;
}
