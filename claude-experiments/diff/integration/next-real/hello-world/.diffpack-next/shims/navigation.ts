// `next/navigation` shim (diffpack next app-router adapter). The client hooks
// (useParams/usePathname/useSearchParams) read the app-router hooks CONTEXTS, fed
// identically by the SSR and client entries — so they resolve on BOTH SSR and the
// browser with no hydration mismatch (NOT window globals, which don't exist during
// SSR). `redirect`/`notFound` on the SERVER throw Next's digest errors, which the
// react-server render's onError captures and turns into a real HTTP redirect / 404;
// on the client they fall back to browser navigation. This module is imported in all
// three graphs, so it uses `React.useContext` (undefined under the react-server
// condition, but the hooks are only ever CALLED inside client components).
import * as React from "react";
import { PathParamsContext, PathnameContext, SearchParamsContext } from "/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/diff/integration/next-real/hello-world/.diffpack-next/hooks-context.ts";

export function useRouter() {
  return {
    push(href) {
      if (typeof window !== "undefined" && typeof window.__diffpack_navigate === "function") {
        window.__diffpack_navigate(href, { replace: false });
      } else if (typeof window !== "undefined") {
        window.location.assign(href);
      }
    },
    replace(href) {
      if (typeof window !== "undefined" && typeof window.__diffpack_navigate === "function") {
        window.__diffpack_navigate(href, { replace: true });
      } else if (typeof window !== "undefined") {
        window.location.replace(href);
      }
    },
    back() { if (typeof window !== "undefined") window.history.back(); },
    forward() { if (typeof window !== "undefined") window.history.forward(); },
    refresh() { if (typeof window !== "undefined") window.location.reload(); },
    prefetch() { /* no-op: this adapter has no prefetch cache */ },
  };
}

export function usePathname() {
  return React.useContext(PathnameContext);
}

export function useSearchParams() {
  return new URLSearchParams(React.useContext(SearchParamsContext) || "");
}

export function useParams() {
  return React.useContext(PathParamsContext) || {};
}

export function redirect(href, type) {
  if (typeof window === "undefined") {
    // Server: throw Next's redirect digest; the react-server render's onError captures
    // it (NEXT_REDIRECT;<type>;<url>;<status>;) and the orchestrator issues a real 307.
    throw Object.assign(new Error("NEXT_REDIRECT"), {
      digest: "NEXT_REDIRECT;" + (type || "replace") + ";" + href + ";307;",
    });
  }
  if (typeof window.__diffpack_navigate === "function") window.__diffpack_navigate(href, { replace: true });
  else window.location.assign(href);
}

export function permanentRedirect(href, type) {
  if (typeof window === "undefined") {
    throw Object.assign(new Error("NEXT_REDIRECT"), {
      digest: "NEXT_REDIRECT;" + (type || "replace") + ";" + href + ";308;",
    });
  }
  if (typeof window.__diffpack_navigate === "function") window.__diffpack_navigate(href, { replace: true });
  else window.location.assign(href);
}

export function notFound() {
  // Both server and client: throw Next's 404 digest. On the server the render's
  // onError captures it and the orchestrator serves the real 404 tree.
  throw Object.assign(new Error("NEXT_HTTP_ERROR_FALLBACK;404"), {
    digest: "NEXT_HTTP_ERROR_FALLBACK;404",
  });
}
