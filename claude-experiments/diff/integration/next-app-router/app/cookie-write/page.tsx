// A Server Component that writes a cookie at the TOP LEVEL (before any Suspense boundary,
// so the write is captured before the streaming shell flushes and rides out as a
// Set-Cookie on the HTML response). Reading/writing next/headers makes this route dynamic,
// so it renders per request. `next build` accepts cookies() in a Server Component.
import { cookies } from "next/headers";

export default async function CookieWritePage() {
  const store = await cookies();
  store.set("sc_page_cookie", "from-server-component", { path: "/", sameSite: "lax" });
  return (
    <main id="cookie-write">
      <h1 id="cw-status">cookie written</h1>
    </main>
  );
}
